"""
Trading Signals Backend
=======================
This server:
1. Connects to Deriv's free WebSocket API to get live OTC price data
2. Runs technical analysis on the prices
3. Generates BUY/SELL signals for 5s, 1min, 3min, 5min timeframes
4. Serves those signals via a REST API that your Lovable web app calls
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
from feed import start_feed, get_latest_ticks
from signals import generate_signals, get_all_signals

# ─── App Startup / Shutdown ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the price feed when the server boots up."""
    print("🚀 Starting price data feed...")
    asyncio.create_task(start_feed())       # runs in background forever
    asyncio.create_task(signal_loop())      # generates signals every 5 seconds
    yield
    print("🛑 Server shutting down.")

app = FastAPI(
    title="Trading Signals API",
    description="Live OTC trading signals for Pocket Option",
    lifespan=lifespan
)

# ─── CORS: Allow your Lovable app to call this API ───────────────────────────
# Replace the lovable URL below with your actual Lovable app URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In production replace * with your Lovable URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Background signal generation loop ───────────────────────────────────────

async def signal_loop():
    """Re-generate signals every 5 seconds based on latest price data."""
    while True:
        try:
            generate_signals()
        except Exception as e:
            print(f"Signal generation error: {e}")
        await asyncio.sleep(5)

# ─── API Routes ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "Trading Signals API is running ✅"}

@app.get("/signals")
def get_signals():
    """
    Returns all current signals.
    Your Lovable app calls this endpoint to show signals on the dashboard.
    
    Example response:
    [
      {
        "asset": "Volatility 75 Index",
        "direction": "BUY",
        "timeframe": "1min",
        "entry_price": 45231.23,
        "confidence": 78,
        "timestamp": "2025-05-02T10:30:00Z",
        "is_vip": false
      },
      ...
    ]
    """
    return get_all_signals()

@app.get("/signals/{timeframe}")
def get_signals_by_timeframe(timeframe: str):
    """
    Get signals filtered by timeframe.
    Valid values: 5s, 1min, 3min, 5min
    """
    all_signals = get_all_signals()
    filtered = [s for s in all_signals if s["timeframe"] == timeframe]
    return filtered

@app.get("/prices")
def get_prices():
    """
    Returns the latest tick price for each asset.
    Useful for showing live prices in your app.
    """
    return get_latest_ticks()

@app.get("/health")
def health():
    """Health check — used by Render/Railway to keep the server alive."""
    ticks = get_latest_ticks()
    return {
        "status": "ok",
        "assets_tracked": len(ticks),
        "prices": ticks
    }