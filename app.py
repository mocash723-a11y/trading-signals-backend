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
    # Load saved thresholds from MongoDB on startup (FIX #3)
    load_adaptive_thresholds_from_mongo()
    
    # Start background tasks
    asyncio.create_task(start_feed())
    asyncio.create_task(signal_loop())
    yield

app = FastAPI(title="Trading Signals API", lifespan=lifespan)

# FIX #10: Lock CORS to your frontend domain in production
# Change this to your actual Lovable frontend URL when deployed
FRONTEND_URL = os.environ.get("FRONTEND_URL", "*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL] if FRONTEND_URL != "*" else ["*"],
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

# ── Auto‑match fallback for missing signal_id ─────────────────────────────
def find_and_resolve_pending_signal(symbol: str, timeframe: str, direction: str, outcome: str):
    """If frontend doesn't send signal_id, match the most recent pending signal."""
    try:
        from feedback import _get_db  # Use the same singleton connection
        db = _get_db()
        if db is None:
            return None
        
        # Find newest pending signal for same symbol, timeframe, direction
        pending = db.pending_signals.find_one(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "direction": direction,
                "outcome": "pending"
            },
            sort=[("timestamp", -1)]
        )
        if pending:
            signal_id = pending["signal_id"]
            update_signal_outcome(signal_id, outcome)
            print(f"Auto-resolved {signal_id} for {symbol} {timeframe} {direction}")
            return signal_id
        return None
    except Exception as e:
        print(f"Auto-match error (non-fatal): {e}")
        return None

# ── Basic endpoints ──────────────────────────────────────────────────────
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

# ── Feedback endpoint (with auto‑match fallback) ─────────────────────────
class FeedbackPayload(BaseModel):
    symbol: str
    timeframe: str
    direction: str
    outcome: str
    confidence: int
    signal_id: Optional[str] = None

@app.post("/feedback")
def submit_feedback(payload: FeedbackPayload):
    # Always record outcome in training_examples (legacy + backup)
    record_outcome(
        symbol=payload.symbol,
        timeframe=payload.timeframe,
        direction=payload.direction,
        outcome=payload.outcome,
        confidence=payload.confidence
    )
    # Try to resolve a pending signal
    if payload.signal_id:
        update_signal_outcome(payload.signal_id, payload.outcome)
    else:
        # Auto-match if signal_id missing (Lovable frontend)
        find_and_resolve_pending_signal(
            payload.symbol, payload.timeframe,
            payload.direction, payload.outcome
        )
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

# Add these new endpoints after your existing ones

from pydantic import BaseModel
from typing import Optional

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
    outcome: str  # "win" or "loss"
    actual_profit_pct: Optional[float] = None
    user_id: Optional[str] = "default"

@app.post("/saved-trades/save")
def save_recommendation_trade(payload: SaveRecommendationTradePayload):
    """Save a trade from recommendations tab with full indicator data."""
    from feedback import save_trade_from_recommendation, _get_db
    from signals import current_signals
    
    indicators = {}
    
    # FIRST: Try to get indicators from memory (fastest)
    cache_key = f"{payload.symbol}_{payload.timeframe}"
    
    if cache_key in current_signals:
        signal = current_signals[cache_key]
        if "indicators" in signal:
            indicators = signal["indicators"]
            print(f"✅ Found indicators in memory for {cache_key}")
    
    # SECOND (FALLBACK): If not in memory, search MongoDB pending_signals
    if not indicators:
        print(f"⚠️ Signal not in memory, searching MongoDB for {cache_key}")
        try:
            db = _get_db()
            if db is not None:
                # Find the most recent pending signal for this symbol/timeframe/direction
                pending = db["pending_signals"].find_one(
                    {
                        "symbol": payload.symbol,
                        "timeframe": payload.timeframe,
                        "direction": payload.direction,
                        "outcome": "pending"  # Only unresolved signals
                    },
                    sort=[("timestamp", -1)]  # Most recent first
                )
                
                if pending and "features" in pending:
                    indicators = pending["features"]
                    print(f"✅ Found indicators in MongoDB pending_signals for {cache_key}")
                else:
                    print(f"❌ No pending signal found in MongoDB for {cache_key}")
        except Exception as e:
            print(f"Fallback MongoDB search error: {e}")
    
    # THIRD: Log if still no indicators found
    if not indicators:
        print(f"⚠️ WARNING: No indicators found for {cache_key} - AI training will be incomplete for this trade")
    
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
        return {
            "status": "saved", 
            "message": "Trade saved!",
            "indicators_found": len(indicators) > 0,
            "source": "memory" if cache_key in current_signals else "mongodb" if indicators else "none"
        }
    return {"status": "error", "message": "Trade already saved or failed"}
    
@app.post("/saved-trades/close")
def close_saved_trade(payload: CloseSavedTradePayload):
    """Record win/loss for a saved trade."""
    from feedback import close_saved_trade
    
    success = close_saved_trade(
        signal_id=payload.signal_id,
        outcome=payload.outcome,
        actual_profit_pct=payload.actual_profit_pct
    )
    
    if success:
        return {"status": "recorded", "message": f"Trade recorded as {payload.outcome.upper()}! Training data updated."}
    return {"status": "error", "message": "Failed to record outcome"}

@app.get("/saved-trades")
def get_saved_trades(user_id: str = "default", status: Optional[str] = None):
    """Get all saved trades for a user."""
    from feedback import get_user_trades
    trades = get_user_trades(user_id, status)
    return {"trades": trades, "count": len(trades)}
