"""
feed.py — Live Price Data Feed
================================
Connects to Deriv's FREE WebSocket API and streams live tick prices
for OTC-style synthetic assets (Volatility Indices).

These are the closest free equivalent to Pocket Option OTC markets.

Deriv Volatility Index → Pocket Option Equivalent:
  Volatility 10 Index   →  Low volatility OTC pairs (EUR/USD OTC)
  Volatility 25 Index   →  Medium volatility OTC pairs (GBP/USD OTC)
  Volatility 50 Index   →  Medium-high OTC pairs (USD/JPY OTC)
  Volatility 75 Index   →  High volatility OTC pairs (most popular)
  Volatility 100 Index  →  Very high volatility OTC

No API key needed for basic tick data — Deriv provides this free.
"""

import asyncio
import json
import websockets
from collections import deque
from datetime import datetime

# ─── Configuration ────────────────────────────────────────────────────────────

# Deriv's free WebSocket endpoint (no account needed)
DERIV_WS_URL = "wss://ws.binaryws.com/websockets/v3?app_id=1089"

# Assets to track — these are Deriv synthetic indices
# They behave like OTC assets (run 24/7, broker-generated prices)
ASSETS = {
    "R_10":  "Volatility 10 Index",
    "R_25":  "Volatility 25 Index",
    "R_50":  "Volatility 50 Index",
    "R_75":  "Volatility 75 Index",
    "R_100": "Volatility 100 Index",
}

# How many ticks to keep in memory per asset (5 min worth at ~1 tick/sec = 300)
TICK_HISTORY_SIZE = 500

# ─── In-memory price storage ──────────────────────────────────────────────────
# tick_data[symbol] = deque of {"price": float, "time": datetime}
tick_data: dict[str, deque] = {
    symbol: deque(maxlen=TICK_HISTORY_SIZE)
    for symbol in ASSETS.keys()
}

# Latest single price per symbol for quick access
latest_price: dict[str, float] = {}

# ─── WebSocket Feed ───────────────────────────────────────────────────────────

async def subscribe_to_asset(ws, symbol: str):
    """Send subscription request to Deriv for tick data on one asset."""
    await ws.send(json.dumps({
        "ticks": symbol,
        "subscribe": 1
    }))
    print(f"  ✅ Subscribed to {ASSETS[symbol]} ({symbol})")

async def start_feed():
    """
    Main feed loop. Connects to Deriv WebSocket and keeps listening.
    Automatically reconnects if the connection drops.
    """
    print("\n📡 Connecting to Deriv price feed...")
    
    while True:  # Outer loop = auto-reconnect on disconnect
        try:
            async with websockets.connect(DERIV_WS_URL) as ws:
                print("✅ Connected to Deriv WebSocket\n")
                
                # Subscribe to all assets
                for symbol in ASSETS.keys():
                    await subscribe_to_asset(ws, symbol)
                
                print("\n📊 Receiving live tick data...\n")
                
                # Listen for incoming tick messages forever
                async for message in ws:
                    process_tick(message)
                    
        except websockets.exceptions.ConnectionClosed:
            print("⚠️  Connection closed. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"⚠️  Feed error: {e}. Reconnecting in 10 seconds...")
            await asyncio.sleep(10)

def process_tick(raw_message: str):
    """Parse a tick message from Deriv and store the price."""
    try:
        msg = json.loads(raw_message)
        
        if msg.get("msg_type") != "tick":
            return  # Ignore non-tick messages (subscription confirmations etc.)
        
        tick = msg["tick"]
        symbol = tick["symbol"]
        price = float(tick["quote"])
        epoch = tick["epoch"]
        
        if symbol not in tick_data:
            return
        
        # Store the tick
        tick_data[symbol].append({
            "price": price,
            "time": datetime.utcfromtimestamp(epoch),
            "epoch": epoch
        })
        latest_price[symbol] = price
        
    except Exception as e:
        pass  # Silently ignore malformed messages

# ─── Public accessor functions ────────────────────────────────────────────────

def get_ticks(symbol: str) -> list:
    """Get all stored ticks for a symbol as a plain list."""
    return list(tick_data.get(symbol, []))

def get_prices(symbol: str) -> list[float]:
    """Get just the price values for a symbol (for indicator calculations)."""
    return [t["price"] for t in tick_data.get(symbol, [])]

def get_latest_ticks() -> dict:
    """Get the current latest price for each asset."""
    return {
        ASSETS[symbol]: {
            "symbol": symbol,
            "price": latest_price.get(symbol, None),
            "ticks_collected": len(tick_data[symbol])
        }
        for symbol in ASSETS.keys()
    }

def get_all_symbols() -> dict:
    """Return the symbol → name mapping."""
    return ASSETS