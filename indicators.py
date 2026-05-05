from datetime import datetime

def ema(prices, period):
    if len(prices) < period:
        return None
    k = 2 / (period + 1)
    val = sum(prices[:period]) / period
    for p in prices[period:]:
        val = p * k + val * (1 - k)
    return round(val, 5)

def ema_series(prices, period):
    if len(prices) < period:
        return []
    k = 2 / (period + 1)
    result = [sum(prices[:period]) / period]
    for p in prices[period:]:
        result.append(p * k + result[-1] * (1 - k))
    return result

def rsi(prices, period=14):
    if len(prices) < period + 1:
        return None
    recent = prices[-(period + 1):]
    gains, losses = [], []
    for i in range(1, len(recent)):
        change = recent[i] - recent[i-1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    if avg_loss == 0:
        return 100.0
    return round(100 - (100 / (1 + avg_gain / avg_loss)), 2)

def macd(prices, fast=12, slow=26, signal=9):
    if len(prices) < slow + signal:
        return None
    fast_ema = ema_series(prices, fast)
    slow_ema = ema_series(prices, slow)
    offset = len(fast_ema) - len(slow_ema)
    macd_line = [f - s for f, s in zip(fast_ema[offset:], slow_ema)]
    if len(macd_line) < signal:
        return None
    signal_series = ema_series(macd_line, signal)
    if not signal_series:
        return None
    current_macd = macd_line[-1]
    current_signal = signal_series[-1]
    cross = "none"
    if len(macd_line) >= 2 and len(signal_series) >= 2:
        if macd_line[-2] <= signal_series[-2] and current_macd > current_signal:
            cross = "bullish"
        elif macd_line[-2] >= signal_series[-2] and current_macd < current_signal:
            cross = "bearish"
    return {
        "macd_line": round(current_macd, 6),
        "signal_line": round(current_signal, 6),
        "histogram": round(current_macd - current_signal, 6),
        "cross": cross
    }

def bollinger_bands(prices, period=20, std_dev=2.0):
    if len(prices) < period:
        return None
    recent = prices[-period:]
    sma = sum(recent) / period
    std = (sum((p - sma)**2 for p in recent) / period) ** 0.5
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    current = prices[-1]
    band_range = upper - lower
    pct_b = ((current - lower) / band_range) if band_range > 0 else 0.5
    return {
        "upper": round(upper, 5),
        "middle": round(sma, 5),
        "lower": round(lower, 5),
        "current": round(current, 5),
        "percent_b": round(pct_b, 3),
        "squeeze": std < (sma * 0.001)
    }

def stochastic(prices, period=14):
    if len(prices) < period:
        return None
    recent = prices[-period:]
    highest, lowest = max(recent), min(recent)
    if highest == lowest:
        return None
    k = ((prices[-1] - lowest) / (highest - lowest)) * 100
    return {"k": round(k, 2), "overbought": k > 80, "oversold": k < 20}

def tick_momentum(prices, lookback=10):
    if len(prices) < lookback:
        return None
    recent = prices[-lookback:]
    ups = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i-1])
    downs = len(recent) - 1 - ups
    direction = "UP" if ups > downs else "DOWN"
    strength = (max(ups, downs) / (len(recent) - 1)) * 100
    consecutive = 0
    for i in range(len(recent) - 1, 0, -1):
        if direction == "UP" and recent[i] > recent[i-1]:
            consecutive += 1
        elif direction == "DOWN" and recent[i] < recent[i-1]:
            consecutive += 1
        else:
            break
    return {
        "direction": direction,
        "strength": round(strength, 1),
        "consecutive": consecutive,
        "price_change": round(recent[-1] - recent[0], 5)
    }

def multi_timeframe_confirm(prices, direction):
    if len(prices) < 200:
        return {"confirmed": False, "bonus": 0, "agreement": "0/3"}
    confirmations = 0
    short_rsi = rsi(prices[-60:], period=7)
    if short_rsi:
        if direction == "BUY" and short_rsi < 50: confirmations += 1
        elif direction == "SELL" and short_rsi > 50: confirmations += 1
    medium_rsi = rsi(prices[-120:], period=10)
    if medium_rsi:
        if direction == "BUY" and medium_rsi < 52: confirmations += 1
        elif direction == "SELL" and medium_rsi > 48: confirmations += 1
    long_rsi = rsi(prices[-240:], period=14)
    if long_rsi:
        if direction == "BUY" and long_rsi < 55: confirmations += 1
        elif direction == "SELL" and long_rsi > 45: confirmations += 1
    bonus = {0: 0, 1: 0, 2: 10, 3: 20}.get(confirmations, 0)
    return {
        "confirmed": confirmations >= 2,
        "bonus": bonus,
        "agreement": f"{confirmations}/3"
    }

def detect_candle_patterns(prices):
    if len(prices) < 30:
        return {"pattern": "none", "bias": "neutral", "bonus": 0}
    candles = []
    for i in range(3):
        start = -(30 - i * 10)
        end = -(20 - i * 10) if (20 - i * 10) > 0 else None
        chunk = prices[start:end]
        if len(chunk) < 5:
            continue
        candles.append({
            "open": chunk[0], "close": chunk[-1],
            "high": max(chunk), "low": min(chunk),
            "body": abs(chunk[-1] - chunk[0]),
            "range": max(chunk) - min(chunk),
            "bullish": chunk[-1] > chunk[0]
        })
    if len(candles) < 2:
        return {"pattern": "none", "bias": "neutral", "bonus": 0}
    prev, curr = candles[-2], candles[-1]
    if curr["range"] > 0:
        body_ratio = curr["body"] / curr["range"]
        if body_ratio < 0.3:
            upper_wick = curr["high"] - max(curr["open"], curr["close"])
            lower_wick = min(curr["open"], curr["close"]) - curr["low"]
            if lower_wick > upper_wick * 2:
                return {"pattern": "Bullish Pin Bar", "bias": "BUY", "bonus": 8}
            elif upper_wick > lower_wick * 2:
                return {"pattern": "Bearish Pin Bar", "bias": "SELL", "bonus": 8}
    if curr["body"] > prev["body"] * 1.5:
        if curr["bullish"] and not prev["bullish"]:
            return {"pattern": "Bullish Engulfing", "bias": "BUY", "bonus": 10}
        elif not curr["bullish"] and prev["bullish"]:
            return {"pattern": "Bearish Engulfing", "bias": "SELL", "bonus": 10}
    return {"pattern": "none", "bias": "neutral", "bonus": 0}

def session_quality(symbol):
    hour = datetime.utcnow().hour
    if symbol.startswith("cry"):
        return {"quality": "good", "multiplier": 1.0, "note": "24/7 crypto market"}
    overlap = 12 <= hour < 16
    london = 7 <= hour < 16
    new_york = 12 <= hour < 20
    dead = hour >= 20 or hour < 6
    if overlap:
        return {"quality": "excellent", "multiplier": 1.15,
                "note": "London + NY overlap — peak accuracy"}
    elif london or new_york:
        return {"quality": "good", "multiplier": 1.0,
                "note": "Active session — good accuracy"}
    elif dead:
        return {"quality": "poor", "multiplier": 0.80,
                "note": "Low liquidity — signals paused"}
    else:
        return {"quality": "moderate", "multiplier": 0.90,
                "note": "Tokyo session — moderate accuracy"}
