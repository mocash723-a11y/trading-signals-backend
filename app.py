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
