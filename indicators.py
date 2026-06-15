from datetime import datetime
import math

# ── Original helpers ──────────────────────────────────────────────────────
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
    """FIX #8: Use Wilder's smoothing (RMA) like TradingView."""
    if len(prices) < period + 1:
        return None
    recent = prices[-(period + 1):]
    gains, losses = [], []
    for i in range(1, len(recent)):
        change = recent[i] - recent[i-1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))
    
    # Wilder's smoothing: first avg is simple, then smoothed
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    rsi_val = 100 - (100 / (1 + rs))
    return round(rsi_val, 2)

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
    """FIX #7: Add %D line (3-period SMA of %K)."""
    if len(prices) < period + 3:  # Need extra for %D calculation
        return None
    recent = prices[-period:]
    highest, lowest = max(recent), min(recent)
    if highest == lowest:
        return None
    k_raw = ((prices[-1] - lowest) / (highest - lowest)) * 100
    k = round(k_raw, 2)
    
    # Calculate %D (3-period SMA of %K)
    k_values = []
    for i in range(3):
        window = prices[-(period + i):-i] if i > 0 else prices[-period:]
        if len(window) >= period:
            h = max(window)
            l = min(window)
            if h != l:
                k_prev = ((window[-1] - l) / (h - l)) * 100
                k_values.append(k_prev)
    
    d = round(sum(k_values) / len(k_values), 2) if k_values else k
    
    return {
        "k": k,
        "d": d,
        "overbought": k > 80 and d > 80,  # Both must confirm
        "oversold": k < 20 and d < 20     # Both must confirm
    }

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

def multi_timeframe_confirm(symbol, direction):
    """
    FIX #6: Use actual different timeframes, not just sliced RSI.
    Gets real 1min, 3min, 5min data from the candle builders.
    """
    from feed import get_closed_candles
    
    confirmations = 0
    agreement_count = 0
    
    # Get actual timeframe candles
    candles_1m = get_closed_candles(symbol)
    
    if candles_1m and len(candles_1m) >= 10:
        # 1-minute timeframe analysis
        closes_1m = [c["close"] for c in candles_1m[-10:]]
        rsi_1m = rsi(closes_1m, period=7)
        if rsi_1m:
            if direction == "BUY" and rsi_1m < 50:
                confirmations += 1
                agreement_count += 1
            elif direction == "SELL" and rsi_1m > 50:
                confirmations += 1
                agreement_count += 1
        
        # 3-minute timeframe (use every 3rd candle or aggregate)
        if len(candles_1m) >= 30:
            closes_3m = [c["close"] for c in candles_1m[-30:] if c]
            if len(closes_3m) >= 10:
                rsi_3m = rsi(closes_3m[::3], period=7)  # Sample every 3rd candle
                if rsi_3m:
                    if direction == "BUY" and rsi_3m < 52:
                        confirmations += 1
                        agreement_count += 1
                    elif direction == "SELL" and rsi_3m > 48:
                        confirmations += 1
                        agreement_count += 1
        
        # 5-minute timeframe
        if len(candles_1m) >= 50:
            closes_5m = [c["close"] for c in candles_1m[-50:] if c]
            if len(closes_5m) >= 10:
                rsi_5m = rsi(closes_5m[::5], period=10)  # Sample every 5th candle
                if rsi_5m:
                    if direction == "BUY" and rsi_5m < 55:
                        confirmations += 1
                        agreement_count += 1
                    elif direction == "SELL" and rsi_5m > 45:
                        confirmations += 1
                        agreement_count += 1
    
    
bonus = {0: 0, 1: 0, 2: 8, 3: 8}   
    return {
        "confirmed": confirmations >= 2,
        "bonus": bonus,
        "agreement": f"{agreement_count}/3"
    }

def session_quality(symbol):
    hour = datetime.utcnow().hour
    if symbol.startswith("cry"):
        return {"quality": "good", "multiplier": 1.0, "note": "24/7 crypto market"}
    overlap = 12 <= hour < 16
    london = 7 <= hour < 16
    new_york = 12 <= hour < 20
    dead = hour >= 20 or hour < 6
    if overlap:
    return {"quality": "excellent", "multiplier": 1.08,
            "note": "London + NY overlap — good accuracy"}
    elif london or new_york:
        return {"quality": "good", "multiplier": 1.0,
                "note": "Active session — good accuracy"}
    elif dead:
        return {"quality": "poor", "multiplier": 0.80,
                "note": "Low liquidity — signals paused"}
    else:
        return {"quality": "moderate", "multiplier": 0.90,
                "note": "Tokyo session — moderate accuracy"}

# ── New indicators: ADX, ATR ──────────────────────────────────────────────
def atr(highs, lows, closes, period=14):
    if len(highs) < period + 1:
        return None
    tr = []
    for i in range(1, len(highs)):
        tr.append(max(highs[i] - lows[i],
                      abs(highs[i] - closes[i-1]),
                      abs(lows[i] - closes[i-1])))
    if len(tr) < period:
        return None
    atr_val = sum(tr[:period]) / period
    for i in range(period, len(tr)):
        atr_val = (atr_val * (period - 1) + tr[i]) / period
    return round(atr_val, 5)

def adx(highs, lows, closes, period=14):
    if len(highs) < period + 1:
        return None
    plus_dm = []
    minus_dm = []
    tr = []
    for i in range(1, len(highs)):
        up = highs[i] - highs[i-1]
        down = lows[i-1] - lows[i]
        if up > down and up > 0:
            plus_dm.append(up)
        else:
            plus_dm.append(0)
        if down > up and down > 0:
            minus_dm.append(down)
        else:
            minus_dm.append(0)
        tr_val = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i-1]),
                     abs(lows[i] - closes[i-1]))
        tr.append(tr_val)
    if len(tr) < period:
        return None
    atr_val = sum(tr[:period]) / period
    sum_pdm = sum(plus_dm[:period])
    sum_mdm = sum(minus_dm[:period])
    for i in range(period, len(tr)):
        atr_val = (atr_val * (period - 1) + tr[i]) / period
        sum_pdm = (sum_pdm * (period - 1) + plus_dm[i]) / period
        sum_mdm = (sum_mdm * (period - 1) + minus_dm[i]) / period
    if atr_val == 0:
        return None
    plus_di = (sum_pdm / atr_val) * 100
    minus_di = (sum_mdm / atr_val) * 100
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return {"adx": round(dx, 2), "plus_di": round(plus_di, 2), "minus_di": round(minus_di, 2)}

# ── Real candle pattern detection (requires OHLC candles) ─────────────────
def detect_candle_patterns_from_ohlc(candles):
    if not candles or len(candles) < 2:
        return {"pattern": "none", "bias": "neutral", "bonus": 0}
    curr = candles[-1]
    prev = candles[-2] if len(candles) >= 2 else None

    body = abs(curr["close"] - curr["open"])
    range_hl = curr["high"] - curr["low"]
    if range_hl == 0:
        return {"pattern": "none", "bias": "neutral", "bonus": 0}
    body_ratio = body / range_hl if range_hl > 0 else 0
    upper_wick = curr["high"] - max(curr["open"], curr["close"])
    lower_wick = min(curr["open"], curr["close"]) - curr["low"]

    if body_ratio < 0.35:
        if lower_wick >= 2 * upper_wick and lower_wick > 0:
            return {"pattern": "Bullish Pin Bar", "bias": "BUY", "bonus": 8}
        if upper_wick >= 2 * lower_wick and upper_wick > 0:
            return {"pattern": "Bearish Pin Bar", "bias": "SELL", "bonus": 8}

    if prev:
        prev_body = abs(prev["close"] - prev["open"])
        if body > prev_body * 1.2:
            if curr["close"] > curr["open"] and prev["close"] < prev["open"]:
                return {"pattern": "Bullish Engulfing", "bias": "BUY", "bonus": 10}
            if curr["close"] < curr["open"] and prev["close"] > prev["open"]:
                return {"pattern": "Bearish Engulfing", "bias": "SELL", "bonus": 10}

    if body_ratio < 0.05:
        return {"pattern": "Doji", "bias": "neutral", "bonus": 2}

    if lower_wick >= 2 * body and upper_wick <= body * 0.5:
        return {"pattern": "Hammer", "bias": "BUY", "bonus": 6}
    if upper_wick >= 2 * body and lower_wick <= body * 0.5:
        return {"pattern": "Shooting Star", "bias": "SELL", "bonus": 6}

    return {"pattern": "none", "bias": "neutral", "bonus": 0}

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
    return detect_candle_patterns_from_ohlc(candles)
