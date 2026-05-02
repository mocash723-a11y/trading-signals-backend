
import asyncio
import json
import websockets
from collections import deque
from datetime import datetime

DERIV_WS_URL = "wss://ws.binaryws.com/websockets/v3?app_id=1089"

ASSETS = {
    "frxEURUSD": "EUR/USD",
    "frxGBPUSD": "GBP/USD",
    "frxUSDJPY": "USD/JPY",
    "frxAUDUSD": "AUD/USD",
    "frxUSDCAD": "USD/CAD",
    "frxEURGBP": "EUR/GBP",
    "frxGBPJPY": "GBP/JPY",
    "frxEURJPY": "EUR/JPY",
    "cryBTCUSD": "Bitcoin/USD",
}

ASSET_GROUPS = {
    "forex": {
        "label": "Forex Pairs",
        "note": "Best accuracy during London & NY sessions (12pm-4pm UTC)",
        "symbols": {
            "frxEURUSD": "EUR/USD",
            "frxGBPUSD": "GBP/USD",
            "frxUSDJPY": "USD/JPY",
            "frxAUDUSD": "AUD/USD",
            "frxUSDCAD": "USD/CAD",
            "frxEURGBP": "EUR/GBP",
            "frxGBPJPY": "GBP/JPY",
            "frxEURJPY": "EUR/JPY",
        }
    },
    "crypto": {
        "label": "Crypto",
        "note": "Trades 24/7. High volatility.",
        "symbols": {
            "cryBTCUSD": "Bitcoin/USD",
        }
    }
}

TICK_HISTORY_SIZE = 600
tick_data = {symbol: deque(maxlen=TICK_HISTORY_SIZE) for symbol in ASSETS}
latest_price = {}
active_symbols = set()

async def subscribe_to_asset(ws, symbol):
    await ws.send(json.dumps({"ticks": symbol, "subscribe": 1}))

async def start_feed():
    print("Connecting to Deriv price feed...")
    while True:
        try:
            async with websockets.connect(DERIV_WS_URL) as ws:
                print("Connected to Deriv WebSocket")
                for symbol in ASSETS:
                    await subscribe_to_asset(ws, symbol)
                    await asyncio.sleep(0.1)
                print(f"Subscribed to {len(ASSETS)} assets")
                async for message in ws:
                    process_tick(message)
        except websockets.exceptions.ConnectionClosed:
            print("Reconnecting in 5s...")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"Error: {e}. Reconnecting in 10s...")
            await asyncio.sleep(10)

def process_tick(raw_message):
    try:
        msg = json.loads(raw_message)
        if msg.get("msg_type") == "tick":
            tick = msg["tick"]
            symbol = tick["symbol"]
            price = float(tick["quote"])
            epoch = tick["epoch"]
            if symbol not in tick_data:
                return
            tick_data[symbol].append({
                "price": price,
                "time": datetime.utcfromtimestamp(epoch),
                "epoch": epoch
            })
            latest_price[symbol] = price
            active_symbols.add(symbol)
    except Exception:
        pass

def get_prices(symbol):
    return [t["price"] for t in tick_data.get(symbol, [])]

def get_latest_ticks():
    return {
        symbol: {
            "name": ASSETS[symbol],
            "price": latest_price.get(symbol),
            "ticks_collected": len(tick_data[symbol]),
            "active": symbol in active_symbols
        }
        for symbol in ASSETS
    }

def get_all_symbols():
    return ASSETS

def get_asset_groups():
    return ASSET_GROUPS

def is_forex_session():
    hour = datetime.utcnow().hour
    return {
        "london":   7  <= hour < 16,
        "new_york": 12 <= hour < 21,
        "tokyo":    0  <= hour < 9,
        "overlap":  12 <= hour < 16,
        "dead_zone": hour >= 21 or hour < 6,
        "current_hour_utc": hour
    }