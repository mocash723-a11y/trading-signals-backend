from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional
import asyncio
from feed import start_feed, get_latest_ticks, get_asset_groups
from signals import (
    generate_signals, get_all_signals,
    get_signal_for, get_recommendations, ml_model
)
from feedback import record_outcome, get_stats, get_pair_accuracy, update_signal_outcome
import joblib
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
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

# ── Feedback endpoints ────────────────────────────────────────────────────────
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
    # If signal_id is provided, update the pending signal for ML
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

# ── Model management ─────────────────────────────────────────────────────────
@app.post("/reload-model")
def reload_model():
    global ml_model
    if os.path.exists("model.pkl"):
        try:
            ml_model = joblib.load("model.pkl")
            return {"status": "model reloaded"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    return {"status": "model not found"}
