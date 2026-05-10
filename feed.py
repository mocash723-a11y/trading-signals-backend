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
        "note": "5min signals only. High volatility.",
        "symbols": {"cryBTCUSD": "Bitcoin/USD"}
    }
}

# ── BUG FIX: Per-pair confidence thresholds restored ─────────────────────────
PAIR_MIN_CONFIDENCE = {
    "frxEURUSD": 62,
    "frxGBPUSD": 64,
    "frxUSDJPY": 64,
    "frxAUDUSD": 65,
    "frxUSDCAD": 65,
    "frxEURGBP": 66,
    "frxGBPJPY": 70,   # Volatile pair — higher bar
    "frxEURJPY": 70,   # Volatile pair — higher bar
    "cryBTCUSD": 78,   # Crypto — only very high confidence
}

# Bitcoin excluded from fast timeframes
BTC_EXCLUDED_TIMEFRAMES = {"5s", "1min", "3min"}

# Signal validity windows (seconds)
SIGNAL_VALIDITY = {"5s": 20, "1min": 45, "3min": 90, "5min": 120}

TICK_HISTORY_SIZE = 600
tick_data = {symbol: deque(maxlen=TICK_HISTORY_SIZE) for symbol in ASSETS}
latest_price = {}
active_symbols = set()

# ── Real OHLC candle builder ──────────────────────────────────────────────────
candle_builders = defaultdict(lambda: {
    "open": None, "high": None, "low": None,
    "close": None, "volume": 0, "start_epoch": None
})
closed_candles = defaultdict(list)
_last_1m_key = defaultdict(str)
_last_5m_key = defaultdict(str)

def _build_key(symbol, epoch, timeframe_minutes):
    dt = datetime.utcfromtimestamp(epoch)
    if timeframe_minutes == 5:
        minute_bucket = (dt.minute // 5) * 5
        dt = dt.replace(minute=minute_bucket, second=0, microsecond=0)
    else:
        dt = dt.replace(second=0, microsecond=0)
    return f"{symbol}_{dt.strftime('%Y%m%d%H%M')}"

def update_candle_for_key(key, price, epoch):
    c = candle_builders[key]
    if c["open"] is None:
        c["open"] = c["high"] = c["low"] = c["close"] = price
        c["volume"] = 1
        c["start_epoch"] = epoch
    else:
        c["high"] = max(c["high"], price)
        c["low"] = min(c["low"], price)
        c["close"] = price
        c["volume"] += 1

def finalise_candle(builder_key):
    candle = candle_builders.pop(builder_key, None)
    if candle and candle["open"] is not None:
        list_key = "_".join(builder_key.split("_")[:-1])
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
            tick_data[symbol].append({
                "price": price,
                "time": datetime.utcfromtimestamp(epoch),
                "epoch": epoch
            })
            latest_price[symbol] = price
            active_symbols.add(symbol)
            # Build 1-min candles
            key_1m = _build_key(symbol, epoch, 1)
            if _last_1m_key[symbol] and key_1m != _last_1m_key[symbol]:
                finalise_candle(_last_1m_key[symbol])
            _last_1m_key[symbol] = key_1m
            update_candle_for_key(key_1m, price, epoch)
            # Build 5-min candles
            key_5m = _build_key(symbol, epoch, 5)
            if _last_5m_key.get(symbol) and key_5m != _last_5m_key[symbol]:
                finalise_candle(_last_5m_key[symbol])
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

def get_signal_validity(timeframe):
    return SIGNAL_VALIDITY.get(timeframe, 60)

def get_pair_min_confidence(symbol):
    return PAIR_MIN_CONFIDENCE.get(symbol, 65)

def get_closed_candles(symbol):
    return closed_candles.get(symbol, [])

def is_forex_session():
    """BUG FIX: Added session_label field that signals.py needs."""
    hour = datetime.utcnow().hour
    overlap  = 12 <= hour < 16
    london   = 7  <= hour < 16
    new_york = 12 <= hour < 20
    dead     = hour >= 20 or hour < 6
    tokyo    = 0  <= hour < 7

    if overlap:
        label = "London + NY Overlap (Best)"
    elif london:
        label = "London Session"
    elif new_york:
        label = "NY Session"
    elif tokyo:
        label = "Tokyo Session"
    else:
        label = "Low Liquidity — Caution"

    return {
        "london":    london,
        "new_york":  new_york,
        "tokyo":     tokyo,
        "overlap":   overlap,
        "dead_zone": dead,
        "current_hour_utc": hour,
        "session_label": label,   # ← BUG FIX: was missing before
    }
