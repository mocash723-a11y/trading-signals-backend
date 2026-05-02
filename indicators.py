"""
indicators.py — Technical Analysis Indicators
==============================================
Calculates RSI, EMA, MACD, and Bollinger Bands
from raw price tick data.

These are all calculated manually (no external library needed)
so they work on ANY Python environment.
"""

from collections import deque

# ─── Exponential Moving Average (EMA) ────────────────────────────────────────

def ema(prices: list[float], period: int) -> float | None:
    """
    Calculate the current EMA value.
    Returns None if not enough data.
    
    EMA gives more weight to recent prices than a simple average.
    """
    if len(prices) < period:
        return None
    
    k = 2 / (period + 1)  # smoothing factor
    
    # Seed with simple average of first `period` prices
    ema_val = sum(prices[:period]) / period
    
    # Apply EMA formula to remaining prices
    for price in prices[period:]:
        ema_val = price * k + ema_val * (1 - k)
    
    return round(ema_val, 5)

def ema_series(prices: list[float], period: int) -> list[float]:
    """Returns full EMA series (needed for MACD)."""
    if len(prices) < period:
        return []
    
    k = 2 / (period + 1)
    result = [sum(prices[:period]) / period]
    
    for price in prices[period:]:
        result.append(price * k + result[-1] * (1 - k))
    
    return result

# ─── RSI (Relative Strength Index) ───────────────────────────────────────────

def rsi(prices: list[float], period: int = 14) -> float | None:
    """
    Calculate RSI.
    
    RSI ranges from 0 to 100:
    - Above 70 = overbought → likely to go DOWN
    - Below 30 = oversold → likely to go UP
    - 50 = neutral
    """
    if len(prices) < period + 1:
        return None
    
    # Use only the most recent prices needed
    recent = prices[-(period + 1):]
    
    gains = []
    losses = []
    
    for i in range(1, len(recent)):
        change = recent[i] - recent[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))
    
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi_val = 100 - (100 / (1 + rs))
    
    return round(rsi_val, 2)

# ─── MACD ─────────────────────────────────────────────────────────────────────

def macd(prices: list[float], 
         fast: int = 12, 
         slow: int = 26, 
         signal: int = 9) -> dict | None:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Returns:
    - macd_line: fast EMA - slow EMA
    - signal_line: 9-period EMA of macd_line
    - histogram: macd_line - signal_line (positive = bullish, negative = bearish)
    - cross: "bullish" if macd just crossed above signal, "bearish" if crossed below
    """
    if len(prices) < slow + signal:
        return None
    
    fast_ema = ema_series(prices, fast)
    slow_ema = ema_series(prices, slow)
    
    # Align the two series (slow_ema is shorter)
    offset = len(fast_ema) - len(slow_ema)
    aligned_fast = fast_ema[offset:]
    
    macd_line = [f - s for f, s in zip(aligned_fast, slow_ema)]
    
    if len(macd_line) < signal:
        return None
    
    signal_line_series = ema_series(macd_line, signal)
    
    if not signal_line_series:
        return None
    
    current_macd = macd_line[-1]
    current_signal = signal_line_series[-1]
    histogram = current_macd - current_signal
    
    # Detect crossover (compare current vs previous bar)
    cross = "none"
    if len(macd_line) >= 2 and len(signal_line_series) >= 2:
        prev_macd = macd_line[-2]
        prev_signal = signal_line_series[-2]
        
        if prev_macd <= prev_signal and current_macd > current_signal:
            cross = "bullish"   # MACD crossed above signal → BUY
        elif prev_macd >= prev_signal and current_macd < current_signal:
            cross = "bearish"   # MACD crossed below signal → SELL
    
    return {
        "macd_line": round(current_macd, 6),
        "signal_line": round(current_signal, 6),
        "histogram": round(histogram, 6),
        "cross": cross
    }

# ─── Bollinger Bands ──────────────────────────────────────────────────────────

def bollinger_bands(prices: list[float], 
                    period: int = 20, 
                    std_dev: float = 2.0) -> dict | None:
    """
    Calculate Bollinger Bands.
    
    - Upper band = SMA + (2 × standard deviation)
    - Lower band = SMA - (2 × standard deviation)
    
    When price touches/breaks lower band → potential BUY
    When price touches/breaks upper band → potential SELL
    """
    if len(prices) < period:
        return None
    
    recent = prices[-period:]
    sma = sum(recent) / period
    
    variance = sum((p - sma) ** 2 for p in recent) / period
    std = variance ** 0.5
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    current = prices[-1]
    
    # Calculate %B (where price is within the bands, 0=lower, 1=upper)
    band_range = upper - lower
    percent_b = ((current - lower) / band_range) if band_range > 0 else 0.5
    
    return {
        "upper": round(upper, 5),
        "middle": round(sma, 5),
        "lower": round(lower, 5),
        "current": round(current, 5),
        "percent_b": round(percent_b, 3),  # < 0.2 = near lower band, > 0.8 = near upper
        "squeeze": std < (sma * 0.001)     # True = low volatility, breakout coming
    }

# ─── Stochastic Oscillator ────────────────────────────────────────────────────

def stochastic(prices: list[float], period: int = 14) -> dict | None:
    """
    Calculate Stochastic Oscillator %K.
    
    - Above 80 = overbought → potential SELL
    - Below 20 = oversold → potential BUY
    """
    if len(prices) < period:
        return None
    
    recent = prices[-period:]
    highest = max(recent)
    lowest = min(recent)
    current = prices[-1]
    
    if highest == lowest:
        return None
    
    k = ((current - lowest) / (highest - lowest)) * 100
    
    return {
        "k": round(k, 2),
        "overbought": k > 80,
        "oversold": k < 20
    }

# ─── Tick Momentum (for 5-second signals) ────────────────────────────────────

def tick_momentum(prices: list[float], lookback: int = 10) -> dict | None:
    """
    Simple momentum for ultra-short 5-second timeframe.
    
    Measures direction and speed of recent price movement.
    NOT a classic indicator — designed for very short timeframes.
    
    Returns:
    - direction: "UP" or "DOWN"
    - strength: 0-100 (how strong the move is)
    - consecutive: how many ticks in a row going same direction
    """
    if len(prices) < lookback:
        return None
    
    recent = prices[-lookback:]
    
    # Count up vs down ticks
    ups = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i-1])
    downs = len(recent) - 1 - ups
    
    # Overall direction
    direction = "UP" if ups > downs else "DOWN"
    strength = (max(ups, downs) / (len(recent) - 1)) * 100
    
    # Consecutive ticks in same direction
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