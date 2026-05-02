"""
signals.py — Signal Generation Engine
=======================================
This is the brain of the system.
It reads live prices from feed.py, runs indicators from indicators.py,
and decides when to generate a BUY or SELL signal.

Signal Strategy by Timeframe:
──────────────────────────────
5 seconds  → Tick momentum (price direction in last 10 ticks)
1 minute   → Fast EMA crossover (5 vs 13) + RSI(7)
3 minutes  → RSI(14) + MACD + Stochastic
5 minutes  → RSI(14) + MACD + Bollinger Bands (full confluence)

Confidence score = how many indicators agree (0-100%)
Minimum confidence to publish signal = 60%
VIP signals = confidence ≥ 80%
"""

from datetime import datetime, timezone
from feed import get_prices, get_all_symbols
from indicators import (
    rsi, ema, macd, bollinger_bands, stochastic, tick_momentum
)

# ─── Configuration ────────────────────────────────────────────────────────────

MIN_CONFIDENCE = 60        # Don't show signals below this confidence %
VIP_CONFIDENCE = 78        # Signals at this confidence+ are marked VIP
MIN_TICKS_REQUIRED = 50    # Need at least this many ticks before generating signals

# ─── In-memory signal storage ─────────────────────────────────────────────────
# Stores the latest signal per (symbol, timeframe) pair
current_signals: dict[str, dict] = {}

# ─── Signal generation for each timeframe ────────────────────────────────────

def signal_5s(symbol: str, name: str, prices: list[float]) -> dict | None:
    """
    5-second signal using tick momentum.
    Requires at least 15 recent ticks.
    """
    if len(prices) < 15:
        return None
    
    momentum = tick_momentum(prices, lookback=12)
    if not momentum:
        return None
    
    # Only signal if momentum is strong enough
    if momentum["strength"] < 65 or momentum["consecutive"] < 4:
        return None
    
    direction = momentum["direction"]
    confidence = min(int(momentum["strength"]), 82)
    
    return _build_signal(
        symbol=symbol,
        name=name,
        direction=direction,
        timeframe="5s",
        confidence=confidence,
        entry_price=prices[-1],
        reason=f"Tick momentum {momentum['consecutive']} consecutive {direction} ticks"
    )

def signal_1min(symbol: str, name: str, prices: list[float]) -> dict | None:
    """
    1-minute signal using fast EMA crossover + RSI.
    
    BUY when: EMA5 crosses above EMA13 AND RSI < 55 (not overbought)
    SELL when: EMA5 crosses below EMA13 AND RSI > 45 (not oversold)
    """
    if len(prices) < 60:
        return None
    
    # Sample prices at ~1-second intervals to build 1-min candles
    # Use last 60 ticks as proxy for 1 minute of data
    sample = prices[-60:]
    
    ema5_now = ema(sample, 5)
    ema13_now = ema(sample, 13)
    ema5_prev = ema(sample[:-3], 5)   # 3 ticks ago
    ema13_prev = ema(sample[:-3], 13)
    rsi_val = rsi(sample, period=7)   # Fast RSI for short timeframe
    
    if None in (ema5_now, ema13_now, ema5_prev, ema13_prev, rsi_val):
        return None
    
    bullish_cross = (ema5_prev <= ema13_prev) and (ema5_now > ema13_now)
    bearish_cross = (ema5_prev >= ema13_prev) and (ema5_now < ema13_now)
    
    if not (bullish_cross or bearish_cross):
        return None
    
    direction = "BUY" if bullish_cross else "SELL"
    
    # RSI confirmation
    rsi_confirms = (direction == "BUY" and rsi_val < 60) or \
                   (direction == "SELL" and rsi_val > 40)
    
    confidence = 65 if not rsi_confirms else 75
    if rsi_confirms and (rsi_val < 40 or rsi_val > 60):
        confidence = 80  # Strong RSI confirmation
    
    return _build_signal(
        symbol=symbol,
        name=name,
        direction=direction,
        timeframe="1min",
        confidence=confidence,
        entry_price=prices[-1],
        reason=f"EMA5/13 {'bullish' if bullish_cross else 'bearish'} cross | RSI {rsi_val}"
    )

def signal_3min(symbol: str, name: str, prices: list[float]) -> dict | None:
    """
    3-minute signal using RSI + MACD + Stochastic.
    
    BUY when: RSI oversold + MACD bullish cross + Stochastic oversold
    SELL when: RSI overbought + MACD bearish cross + Stochastic overbought
    """
    if len(prices) < 150:
        return None
    
    sample = prices[-180:]  # Last 3 minutes worth of ticks
    
    rsi_val = rsi(sample, period=14)
    macd_val = macd(sample, fast=8, slow=17, signal=9)  # Faster MACD for shorter TF
    stoch = stochastic(sample, period=14)
    
    if not all([rsi_val, macd_val, stoch]):
        return None
    
    bullish_signals = 0
    bearish_signals = 0
    
    # RSI
    if rsi_val < 35:
        bullish_signals += 1
    elif rsi_val > 65:
        bearish_signals += 1
    
    # MACD
    if macd_val["cross"] == "bullish" or macd_val["histogram"] > 0:
        bullish_signals += 1
    elif macd_val["cross"] == "bearish" or macd_val["histogram"] < 0:
        bearish_signals += 1
    
    # Stochastic
    if stoch["oversold"]:
        bullish_signals += 1
    elif stoch["overbought"]:
        bearish_signals += 1
    
    # Need at least 2 out of 3 indicators to agree
    if bullish_signals >= 2:
        direction = "BUY"
        confidence = 65 + (bullish_signals * 8)
    elif bearish_signals >= 2:
        direction = "SELL"
        confidence = 65 + (bearish_signals * 8)
    else:
        return None  # No consensus
    
    return _build_signal(
        symbol=symbol,
        name=name,
        direction=direction,
        timeframe="3min",
        confidence=min(confidence, 88),
        entry_price=prices[-1],
        reason=f"RSI:{rsi_val} | MACD:{macd_val['cross']} | Stoch:{round(stoch['k'],1)}"
    )

def signal_5min(symbol: str, name: str, prices: list[float]) -> dict | None:
    """
    5-minute signal using full confluence: RSI + MACD + Bollinger Bands.
    This is the most reliable signal type.
    
    BUY: RSI oversold + price at/below lower BB + MACD histogram turning positive
    SELL: RSI overbought + price at/above upper BB + MACD histogram turning negative
    """
    if len(prices) < 250:
        return None
    
    sample = prices[-300:]  # 5 minutes of data
    
    rsi_val = rsi(sample, period=14)
    macd_val = macd(sample, fast=12, slow=26, signal=9)
    bb = bollinger_bands(sample, period=20, std_dev=2.0)
    
    if not all([rsi_val, macd_val, bb]):
        return None
    
    bullish_signals = 0
    bearish_signals = 0
    reasons = []
    
    # RSI
    if rsi_val < 35:
        bullish_signals += 1
        reasons.append(f"RSI oversold ({rsi_val})")
    elif rsi_val > 65:
        bearish_signals += 1
        reasons.append(f"RSI overbought ({rsi_val})")
    
    # Bollinger Bands
    if bb["percent_b"] < 0.25:
        bullish_signals += 1
        reasons.append("Price near lower BB")
    elif bb["percent_b"] > 0.75:
        bearish_signals += 1
        reasons.append("Price near upper BB")
    
    # MACD
    if macd_val["cross"] == "bullish":
        bullish_signals += 2  # Cross = strong signal, worth 2 points
        reasons.append("MACD bullish cross")
    elif macd_val["histogram"] > 0:
        bullish_signals += 1
        reasons.append("MACD positive")
    elif macd_val["cross"] == "bearish":
        bearish_signals += 2
        reasons.append("MACD bearish cross")
    elif macd_val["histogram"] < 0:
        bearish_signals += 1
        reasons.append("MACD negative")
    
    # Require strong confluence for 5min signals
    if bullish_signals >= 3:
        direction = "BUY"
        confidence = 70 + min(bullish_signals * 5, 20)
    elif bearish_signals >= 3:
        direction = "SELL"
        confidence = 70 + min(bearish_signals * 5, 20)
    else:
        return None
    
    return _build_signal(
        symbol=symbol,
        name=name,
        direction=direction,
        timeframe="5min",
        confidence=min(confidence, 91),
        entry_price=prices[-1],
        reason=" | ".join(reasons)
    )

# ─── Signal builder helper ────────────────────────────────────────────────────

def _build_signal(symbol, name, direction, timeframe, confidence, entry_price, reason) -> dict | None:
    """Create a standardized signal dict."""
    if confidence < MIN_CONFIDENCE:
        return None
    
    return {
        "id": f"{symbol}_{timeframe}_{int(datetime.now().timestamp())}",
        "asset": name,
        "symbol": symbol,
        "direction": direction,         # "BUY" or "SELL"
        "timeframe": timeframe,         # "5s", "1min", "3min", "5min"
        "entry_price": round(entry_price, 5),
        "confidence": confidence,        # 0-100
        "is_vip": confidence >= VIP_CONFIDENCE,
        "reason": reason,               # Why the signal was generated
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

# ─── Main generation function ─────────────────────────────────────────────────

def generate_signals():
    """
    Called every 5 seconds by the background loop in app.py.
    Iterates through all assets, runs all timeframe strategies,
    and updates current_signals.
    """
    symbols = get_all_symbols()  # {symbol: name}
    
    for symbol, name in symbols.items():
        prices = get_prices(symbol)
        
        if len(prices) < MIN_TICKS_REQUIRED:
            continue  # Not enough data yet
        
        # Run all timeframe strategies
        for strategy_fn in [signal_5s, signal_1min, signal_3min, signal_5min]:
            signal = strategy_fn(symbol, name, prices)
            if signal:
                key = f"{symbol}_{signal['timeframe']}"
                current_signals[key] = signal

def get_all_signals() -> list[dict]:
    """Return all current active signals, sorted by confidence descending."""
    signals = list(current_signals.values())
    
    # Sort: VIP first, then by confidence
    signals.sort(key=lambda s: (s["is_vip"], s["confidence"]), reverse=True)
    
    return signals