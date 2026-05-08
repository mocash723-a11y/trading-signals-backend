import asyncio
import json
import websockets
from collections import deque, defaultdict
from datetime import datetime, timezone

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

# ── OHLC candle builder ─────────────────────────────────────────────────────
candle_builders = defaultdict(lambda: {"open": None, "high": None, "low": None, "close": None, "volume": 0, "start_epoch": None})
closed_candles = defaultdict(list)   # { "symbol_1min": [candle, ...], ... }

def _build_key(symbol, epoch, timeframe_minutes):
    dt = datetime.utcfromtimestamp(epoch)
    if timeframe_minutes == 5:
        minute_bucket = (dt.minute // 5) * 5
        dt = dt.replace(minute=minute_bucket, second=0, microsecond=0)
    else:  # 1 minute
        dt = dt.replace(second=0, microsecond=0)
    return f"{symbol}_{dt.strftime('%Y%m%d%H%M')}"

def update_candle_for_key(key, price, epoch):
    c = candle_builders[key]
    if c["open"] is None:
        c["open"] = price
        c["high"] = price
        c["low"] = price
        c["close"] = price
        c["volume"] = 1
        c["start_epoch"] = epoch
    else:
        c["high"] = max(c["high"], price)
        c["low"] = min(c["low"], price)
        c["close"] = price
        c["volume"] += 1

_last_1m_key = defaultdict(str)
_last_5m_key = defaultdict(str)

def finalise_candle(builder_key, tf_minutes):
    candle = candle_builders.pop(builder_key, None)
    if candle and candle["open"] is not None:
        list_key = builder_key.rsplit("_", 1)[0]  # "symbol_YYYYMMDDHHMM"
        candles = closed_candles[list_key]
        candles.append(candle)
        if len(candles) > 200:
            candles.pop(0)

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

            # Store raw tick
            tick_data[symbol].append({
                "price": price,
                "time": datetime.utcfromtimestamp(epoch),
                "epoch": epoch
            })
            latest_price[symbol] = price
            active_symbols.add(symbol)

            # Build 1‑min candles
            key_1m = _build_key(symbol, epoch, 1)
            if _last_1m_key[symbol] and key_1m != _last_1m_key[symbol]:
                finalise_candle(_last_1m_key[symbol], 1)
            _last_1m_key[symbol] = key_1m
            update_candle_for_key(key_1m, price, epoch)

            # Build 5‑min candles
            key_5m = _build_key(symbol, epoch, 5)
            if _last_5m_key.get(symbol) and key_5m != _last_5m_key[symbol]:
                finalise_candle(_last_5m_key[symbol], 5)
            _last_5m_key[symbol] = key_5m
            update_candle_for_key(key_5m, price, epoch)
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

# ── Missing functions that signals.py expects ─────────────────────────────
SIGNAL_VALIDITY = {"5s": 30, "1min": 60, "3min": 180, "5min": 300}
BTC_EXCLUDED_TIMEFRAMES = {"5s", "1min", "3min"}

def get_signal_validity(timeframe):
    return SIGNAL_VALIDITY.get(timeframe, 60)

def get_pair_min_confidence(symbol):
    return 62

def get_closed_candles(symbol):
    """Return list of dict (open/high/low/close/volume/start_epoch) for last 200 candles of symbol."""
    return closed_candles.get(symbol, [])
