
from datetime import datetime, timezone
from feed import get_prices, get_all_symbols
from indicators import (
    rsi, ema, macd, bollinger_bands, stochastic, tick_momentum,
    multi_timeframe_confirm, detect_candle_patterns, session_quality
)

MIN_CONFIDENCE = 62
VIP_CONFIDENCE = 78
MIN_TICKS = 60

current_signals = {}

def _build_signal(symbol, name, direction, timeframe, confidence, entry_price, reason, extras=None):
    if confidence < MIN_CONFIDENCE:
        return None
    signal = {
        "id": f"{symbol}_{timeframe}_{int(datetime.now().timestamp())}",
        "asset": name,
        "symbol": symbol,
        "direction": direction,
        "timeframe": timeframe,
        "entry_price": round(entry_price, 5),
        "confidence": min(int(confidence), 93),
        "is_vip": confidence >= VIP_CONFIDENCE,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if extras:
        signal.update(extras)
    return signal

def _apply_boosters(symbol, prices, direction, base_confidence, reasons):
    confidence = base_confidence
    mtf = multi_timeframe_confirm(prices, direction)
    if mtf["confirmed"]:
        confidence += mtf["bonus"]
        reasons.append(f"Multi-TF confirmed ({mtf['agreement']} agree)")
    candle = detect_candle_patterns(prices)
    if candle["bias"] == direction and candle["bonus"] > 0:
        confidence += candle["bonus"]
        reasons.append(f"{candle['pattern']} detected")
    session = session_quality(symbol)
    confidence *= session["multiplier"]
    if session["quality"] == "excellent":
        reasons.append("Peak session (London+NY overlap)")
    elif session["quality"] == "poor":
        reasons.append("Low liquidity — reduce position size")
    return confidence

def signal_5s(symbol, name, prices):
    if len(prices) < 20:
        return None
    momentum = tick_momentum(prices, lookback=15)
    if not momentum or momentum["strength"] < 68 or momentum["consecutive"] < 5:
        return None
    direction = momentum["direction"]
    reasons = [f"{momentum['consecutive']} consecutive {direction} ticks"]
    confidence = _apply_boosters(symbol, prices, direction, momentum["strength"] * 0.85, reasons)
    return _build_signal(symbol, name, direction, "5s", confidence, prices[-1], " | ".join(reasons))

def signal_1min(symbol, name, prices):
    if len(prices) < 70:
        return None
    sample = prices[-70:]
    ema5_now = ema(sample, 5)
    ema13_now = ema(sample, 13)
    ema5_prev = ema(sample[:-5], 5)
    ema13_prev = ema(sample[:-5], 13)
    rsi_val = rsi(sample, period=7)
    if None in (ema5_now, ema13_now, ema5_prev, ema13_prev, rsi_val):
        return None
    bullish = ema5_prev <= ema13_prev and ema5_now > ema13_now
    bearish = ema5_prev >= ema13_prev and ema5_now < ema13_now
    if not (bullish or bearish):
        return None
    direction = "BUY" if bullish else "SELL"
    base = 65
    reasons = [f"EMA5/13 {'bullish' if bullish else 'bearish'} cross", f"RSI {rsi_val}"]
    if (direction == "BUY" and rsi_val < 45) or (direction == "SELL" and rsi_val > 55):
        base = 72
        reasons.append("RSI confirms")
    confidence = _apply_boosters(symbol, prices, direction, base, reasons)
    return _build_signal(symbol, name, direction, "1min", confidence, prices[-1], " | ".join(reasons))

def signal_3min(symbol, name, prices):
    if len(prices) < 160:
        return None
    sample = prices[-180:]
    rsi_val = rsi(sample, period=14)
    macd_val = macd(sample, fast=8, slow=17, signal=9)
    stoch = stochastic(sample, period=14)
    if not all([rsi_val, macd_val, stoch]):
        return None
    bull, bear, reasons = 0, 0, []
    if rsi_val < 35: bull += 1; reasons.append(f"RSI oversold ({rsi_val})")
    elif rsi_val > 65: bear += 1; reasons.append(f"RSI overbought ({rsi_val})")
    if macd_val["cross"] == "bullish": bull += 1; reasons.append("MACD bullish cross")
    elif macd_val["cross"] == "bearish": bear += 1; reasons.append("MACD bearish cross")
    elif macd_val["histogram"] > 0: bull += 1
    elif macd_val["histogram"] < 0: bear += 1
    if stoch["oversold"]: bull += 1; reasons.append("Stochastic oversold")
    elif stoch["overbought"]: bear += 1; reasons.append("Stochastic overbought")
    if bull >= 2:
        confidence = _apply_boosters(symbol, prices, "BUY", 63 + bull * 7, reasons)
        return _build_signal(symbol, name, "BUY", "3min", confidence, prices[-1], " | ".join(reasons))
    elif bear >= 2:
        confidence = _apply_boosters(symbol, prices, "SELL", 63 + bear * 7, reasons)
        return _build_signal(symbol, name, "SELL", "3min", confidence, prices[-1], " | ".join(reasons))
    return None

def signal_5min(symbol, name, prices):
    if len(prices) < 260:
        return None
    sample = prices[-300:]
    rsi_val = rsi(sample, period=14)
    macd_val = macd(sample, fast=12, slow=26, signal=9)
    bb = bollinger_bands(sample, period=20, std_dev=2.0)
    if not all([rsi_val, macd_val, bb]):
        return None
    bull, bear, reasons = 0, 0, []
    if rsi_val < 35: bull += 1; reasons.append(f"RSI oversold ({rsi_val})")
    elif rsi_val > 65: bear += 1; reasons.append(f"RSI overbought ({rsi_val})")
    if bb["percent_b"] < 0.25: bull += 1; reasons.append("Price near lower BB")
    elif bb["percent_b"] > 0.75: bear += 1; reasons.append("Price near upper BB")
    if macd_val["cross"] == "bullish": bull += 2; reasons.append("MACD bullish cross")
    elif macd_val["histogram"] > 0: bull += 1; reasons.append("MACD positive")
    elif macd_val["cross"] == "bearish": bear += 2; reasons.append("MACD bearish cross")
    elif macd_val["histogram"] < 0: bear += 1; reasons.append("MACD negative")
    if bull >= 3:
        confidence = _apply_boosters(symbol, prices, "BUY", 68 + min(bull * 5, 18), reasons)
        return _build_signal(symbol, name, "BUY", "5min", confidence, prices[-1], " | ".join(reasons))
    elif bear >= 3:
        confidence = _apply_boosters(symbol, prices, "SELL", 68 + min(bear * 5, 18), reasons)
        return _build_signal(symbol, name, "SELL", "5min", confidence, prices[-1], " | ".join(reasons))
    return None

STRATEGY_MAP = {"5s": signal_5s, "1min": signal_1min, "3min": signal_3min, "5min": signal_5min}

def generate_signals():
    for symbol, name in get_all_symbols().items():
        prices = get_prices(symbol)
        if len(prices) < MIN_TICKS:
            continue
        for tf, fn in STRATEGY_MAP.items():
            try:
                signal = fn(symbol, name, prices)
                if signal:
                    current_signals[f"{symbol}_{tf}"] = signal
            except Exception:
                pass

def get_all_signals():
    signals = list(current_signals.values())
    signals.sort(key=lambda s: (s["is_vip"], s["confidence"]), reverse=True)
    return signals

def get_signal_for(symbol, timeframe):
    name = get_all_symbols().get(symbol, symbol)
    prices = get_prices(symbol)
    if len(prices) < MIN_TICKS:
        return None
    fn = STRATEGY_MAP.get(timeframe)
    if not fn:
        return None
    fresh = fn(symbol, name, prices)
    if fresh:
        current_signals[f"{symbol}_{timeframe}"] = fresh
        return fresh
    return current_signals.get(f"{symbol}_{timeframe}")

def get_recommendations():
    all_sigs = get_all_signals()
    if not all_sigs:
        return {
            "status": "loading",
            "message": "Collecting price data... check back in 2 minutes.",
            "recommendations": []
        }
    top3 = all_sigs[:3]
    rank_labels = ["Best Trade Now", "Second Best", "Third Best"]
    recommendations = []
    for i, sig in enumerate(top3):
        session = session_quality(sig["symbol"])
        recommendations.append({
            "rank": i + 1,
            "rank_label": rank_labels[i],
            "asset": sig["asset"],
            "symbol": sig["symbol"],
            "direction": sig["direction"],
            "timeframe": sig["timeframe"],
            "confidence": sig["confidence"],
            "entry_price": sig["entry_price"],
            "is_vip": sig["is_vip"],
            "session_quality": session["quality"],
            "session_note": session["note"],
            "why": _explain(sig, session),
            "timestamp": sig["timestamp"],
        })
    return {
        "status": "ok",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "recommendations": recommendations
    }

def _explain(sig, session):
    direction_text = "BUY — price going UP" if sig["direction"] == "BUY" else "SELL — price going DOWN"
    tf_map = {"5s": "5 seconds", "1min": "1 minute", "3min": "3 minutes", "5min": "5 minutes"}
    strength = "Very strong" if sig["confidence"] >= 85 else "Strong" if sig["confidence"] >= 75 else "Moderate"
    return (
        f"{strength} {direction_text} on {sig['asset']}. "
        f"Set expiry to {tf_map.get(sig['timeframe'], sig['timeframe'])}. "
        f"Confidence: {sig['confidence']}%. "
        f"Market: {session['note']}. "
        f"Indicators: {sig['reason']}."
    )