from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
import asyncio
from feed import start_feed, get_latest_ticks, get_asset_groups
from signals import (
    generate_signals, get_all_signals,
    get_signal_for, get_recommendations
)
from feedback import record_outcome, get_stats, get_pair_accuracy

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

# ── Basic endpoints ───────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "Trading Signals API is running"}

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
    symbol: str        # e.g. "frxEURUSD"
    timeframe: str     # e.g. "1min"
    direction: str     # "BUY" or "SELL"
    outcome: str       # "win" or "loss"
    confidence: int    # confidence score at time of signal

@app.post("/feedback")
def submit_feedback(payload: FeedbackPayload):
    """
    Called by the app when user marks a trade as WIN or LOSS.
    Stores the result and uses it to improve future signals.
    """
    record_outcome(
        symbol=payload.symbol,
        timeframe=payload.timeframe,
        direction=payload.direction,
        outcome=payload.outcome,
        confidence=payload.confidence
    )
    stats = get_pair_accuracy(payload.symbol, payload.timeframe)
    return {
        "status": "recorded",
        "message": f"Thanks! {payload.symbol} {payload.timeframe} accuracy: {stats.get('win_rate', 'N/A')}%",
        "pair_stats": stats
    }

@app.get("/stats")
def get_all_stats():
    """
    Returns win/loss statistics for all pairs and timeframes.
    Used by the app to show your personal accuracy dashboard.
    """
    return get_stats()

@app.get("/stats/{symbol}/{timeframe}")
def get_pair_stats(symbol: str, timeframe: str):
    return get_pair_accuracy(symbol, timeframe)
