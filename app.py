from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional
import asyncio
import os
from feed import start_feed, get_latest_ticks, get_asset_groups
from signals import (
    generate_signals, get_all_signals,
    get_signal_for, get_recommendations, ml_model
)
from feedback import record_outcome, get_stats, get_pair_accuracy, update_signal_outcome, load_adaptive_thresholds_from_mongo

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_adaptive_thresholds_from_mongo()
    asyncio.create_task(start_feed())
    asyncio.create_task(signal_loop())
    yield

app = FastAPI(title="Trading Signals API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

async def signal_loop():
    while True:
        try:
            generate_signals()
        except Exception as e:
            print(f"Signal error: {e}")
        await asyncio.sleep(5)

# ==================== BASIC ENDPOINTS ====================

@app.get("/")
def root():
    return {"status": "Trading Signals API is running", "ml_loaded": ml_model is not None}

@app.get("/health")
def health():
    return {"status": "ok", "prices": get_latest_ticks()}

@app.get("/assets")
def assets():
    return get_asset_groups()

@app.get("/signals")
def get_signals():
    return get_all_signals()

@app.get("/signals/{timeframe}")
def get_signals_by_timeframe(timeframe: str):
    return [s for s in get_all_signals() if s["timeframe"] == timeframe]

@app.get("/signal/{symbol}/{timeframe}")
def get_specific_signal(symbol: str, timeframe: str):
    result = get_signal_for(symbol, timeframe)
    if result:
        return result
    return {
        "status": "no_signal",
        "message": "No clear signal right now. Market is consolidating. Try again in 30 seconds.",
        "symbol": symbol,
        "timeframe": timeframe
    }

@app.get("/recommend")
def recommend():
    return get_recommendations()

@app.get("/prices")
def get_prices():
    return get_latest_ticks()

# ==================== FEEDBACK ====================

class FeedbackPayload(BaseModel):
    symbol: str
    timeframe: str
    direction: str
    outcome: str
    confidence: int
    signal_id: Optional[str] = None

@app.post("/feedback")
def submit_feedback(payload: FeedbackPayload):
    record_outcome(
        symbol=payload.symbol,
        timeframe=payload.timeframe,
        direction=payload.direction,
        outcome=payload.outcome,
        confidence=payload.confidence
    )
    if payload.signal_id:
        update_signal_outcome(payload.signal_id, payload.outcome)
    stats = get_pair_accuracy(payload.symbol, payload.timeframe)
    return {
        "status": "recorded",
        "message": f"Thanks! {payload.symbol} {payload.timeframe} accuracy: {stats.get('win_rate', 'N/A')}%",
        "pair_stats": stats
    }

@app.get("/stats")
def get_all_stats():
    return get_stats()

@app.get("/stats/{symbol}/{timeframe}")
def get_pair_stats(symbol: str, timeframe: str):
    return get_pair_accuracy(symbol, timeframe)

@app.post("/reload-model")
def reload_model():
    return {"status": "Model reload feature will be activated after you upload model.pkl"}

# ==================== SAVED TRADES (Recommendations) ====================

class SaveRecommendationTradePayload(BaseModel):
    signal_id: str
    symbol: str
    timeframe: str
    direction: str
    entry_price: float
    confidence: int
    user_id: Optional[str] = "default"

class CloseSavedTradePayload(BaseModel):
    signal_id: str
    outcome: str
    actual_profit_pct: Optional[float] = None
    user_id: Optional[str] = "default"

@app.post("/saved-trades/save")
def save_recommendation_trade(payload: SaveRecommendationTradePayload):
    from feedback import save_trade_from_recommendation
    from signals import current_signals

    indicators = {}
    cache_key = f"{payload.symbol}_{payload.timeframe}"
    if cache_key in current_signals:
        signal = current_signals[cache_key]
        if "indicators" in signal:
            indicators = signal["indicators"]
            print(f"Captured indicators for {cache_key}: {list(indicators.keys())}")
        else:
            print(f"Warning: Signal {cache_key} has no 'indicators' field")
    else:
        print(f"Warning: No current signal for {cache_key}")

    success = save_trade_from_recommendation(
        signal_id=payload.signal_id,
        symbol=payload.symbol,
        timeframe=payload.timeframe,
        direction=payload.direction,
        entry_price=payload.entry_price,
        confidence=payload.confidence,
        indicators=indicators,
        user_id=payload.user_id
    )
    if success:
        return {"status": "saved", "message": "Trade saved!", "indicators_captured": len(indicators) > 0}
    return {"status": "error", "message": "Trade already saved or failed"}

@app.post("/saved-trades/close")
def close_saved_trade(payload: CloseSavedTradePayload):
    from feedback import close_saved_trade
    success = close_saved_trade(payload.signal_id, payload.outcome, payload.actual_profit_pct)
    if success:
        return {"status": "recorded", "message": f"Trade recorded as {payload.outcome.upper()}!"}
    return {"status": "error", "message": "Failed to record outcome"}

@app.get("/saved-trades")
def get_saved_trades(user_id: str = "default", status: Optional[str] = None):
    from feedback import get_user_trades
    trades = get_user_trades(user_id, status)
    return {"trades": trades, "count": len(trades)}

# ==================== AI TRAINING STATS ====================

@app.get("/ai-stats")
def get_ai_stats():
    import json
    from pymongo import MongoClient

    MONGO_URI = os.environ.get("MONGO_URI")
    if not MONGO_URI:
        return {"status": "error", "message": "MONGO_URI not configured"}

    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client["trading_signals"]

        total_trades = db.training_data.count_documents({})
        wins = db.training_data.count_documents({"outcome": "win"})
        losses = db.training_data.count_documents({"outcome": "loss"})

        model_exists = os.path.exists("model.pkl")
        model_metadata = None
        if os.path.exists("model_metadata.json"):
            with open("model_metadata.json", "r") as f:
                model_metadata = json.load(f)

        if total_trades >= 500:
            status = "excellent"
            msg = "✅ Excellent! Model is highly accurate."
        elif total_trades >= 200:
            status = "good"
            msg = "👍 Good data. Model learning patterns."
        elif total_trades >= 100:
            status = "decent"
            msg = "📈 Decent data. Keep trading."
        elif total_trades >= 50:
            status = "learning"
            msg = "🔄 Learning. Need 50+ more trades."
        else:
            status = "needs_data"
            msg = f"📊 Need {50 - total_trades} more trades to begin training."

        return {
            "status": status,
            "message": msg,
            "model_exists": model_exists,
            "training_data": {
                "total_trades": total_trades,
                "wins": wins,
                "losses": losses,
                "win_rate": round(wins / total_trades * 100, 1) if total_trades else 0
            },
            "model_metadata": model_metadata,
            "requirements": {"minimum_trades": 50, "recommended_trades": 200, "excellent_trades": 500}
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/train-ai")
def train_ai_manual():
    import subprocess
    import threading

    def run_training():
        try:
            result = subprocess.run(["python", "train_model.py"], capture_output=True, text=True, timeout=300)
            print(f"Training output: {result.stdout}")
            if result.stderr:
                print(f"Training errors: {result.stderr}")
        except Exception as e:
            print(f"Training failed: {e}")

    threading.Thread(target=run_training).start()
    return {"status": "training_started", "message": "AI training has started. Check back in 1-2 minutes."}

@app.get("/debug/verify-signal/{signal_id}")
def verify_signal(signal_id: str):
    from feedback import _get_db
    db = _get_db()
    if not db:
        return {"error": "DB not available"}
    pending = db.pending_signals.find_one({"signal_id": signal_id})
    if pending:
        return {
            "exists": True,
            "has_features": bool(pending.get("features")),
            "features_keys": list(pending.get("features", {}).keys())
        }
    return {"exists": False, "message": "Signal ID not found in pending_signals"}
