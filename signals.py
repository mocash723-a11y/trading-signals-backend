from datetime import datetime, timezone
from feed import (
    get_prices, get_all_symbols, get_signal_validity,
    get_pair_min_confidence, BTC_EXCLUDED_TIMEFRAMES,
    GOLD_EXCLUDED_TIMEFRAMES, is_forex_session,
    get_closed_candles, is_gold_session
)
from indicators import (
    rsi, ema, macd, bollinger_bands, stochastic, tick_momentum,
    multi_timeframe_confirm, detect_candle_patterns, session_quality,
    adx, detect_candle_patterns_from_ohlc
)
from feedback import get_adaptive_threshold, save_pending_signal

ml_model = None
current_signals = {}

def _get_min_confidence(symbol, timeframe):
    pair_default = get_pair_min_confidence(symbol)
    adaptive = get_adaptive_threshold(symbol, timeframe)
    return max(pair_default, adaptive)

def _build_signal(symbol, name, direction, timeframe, confidence, entry_price, reason, extras=None):
    min_conf = _get_min_confidence(symbol, timeframe)
    if confidence < min_conf:
        return None
    validity_seconds = get_signal_validity(timeframe)
    signal = {
        "id": f"{symbol}_{timeframe}_{int(datetime.now().timestamp())}",
        "asset": name,
        "symbol": symbol,
        "direction": direction,
        "timeframe": timeframe,
        "entry_price": round(entry_price, 5),
        "confidence": min(int(confidence), 93),
        "is_vip": confidence >= 78,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "valid_for_seconds": validity_seconds,
        "valid_until": datetime.fromtimestamp(
            datetime.now().timestamp() + validity_seconds, tz=timezone.utc
        ).isoformat(),
    }
    if extras:
        signal.update(extras)
    return signal

def _apply_boosters(symbol, prices, direction, base_confidence, reasons, candles_1m=None):
    confidence = base_confidence
    mtf = multi_timeframe_confirm(prices, direction)
    if mtf["confirmed"]:
        confidence += mtf["bonus"]
        reasons.append(f"Multi-TF confirmed ({mtf['agreement']} agree)")
    if candles_1m and len(candles_1m) >= 2:
        candle = detect_candle_patterns_from_ohlc(candles_1m)
    else:
        candle = detect_candle_patterns(prices)
    if candle and candle["bias"] == direction and candle["bonus"] > 0:
        confidence += candle["bonus"]
        reasons.append(f"{candle['pattern']} detected")
    session = session_quality(symbol)
    confidence *= session["multiplier"]
    if session["quality"] == "excellent":
        reasons.append("Peak session")
    elif session["quality"] == "poor":
        reasons.append("Low liquidity")
    if not symbol.startswith("cry") and candles_1m and len(candles_1m) >= 15:
        highs  = [c["high"]  for c in candles_1m[-15:]]
        lows   = [c["low"]   for c in candles_1m[-15:]]
        closes = [c["close"] for c in candles_1m[-15:]]
        adx_val = adx(highs, lows, closes, period=14)
        if adx_val and adx_val["adx"] < 20:
            confidence *= 0.9
            reasons.append("Low ADX — ranging market")
    return confidence

def _session_allows_signal(symbol):
    """Check if current session allows signals for this symbol."""
    if symbol.startswith("cry"):
        return True  # Bitcoin always allowed
    if symbol == "frxXAUUSD":
        return is_gold_session()["active"]  # Gold — check gold session
    return not is_forex_session()["dead_zone"]  # Forex — check dead zone

# ── Timeframe strategies ──────────────────────────────────────────────────────

def signal_5s(symbol, name, prices):
    # Bitcoin and Gold excluded from 5s
    if symbol == "cryBTCUSD" or symbol in GOLD_EXCLUDED_TIMEFRAMES:
        return None
    if symbol == "frxXAUUSD":
        return None
    if not _session_allows_signal(symbol):
        return None
    if len(prices) < 20:
        return None
    momentum = tick_momentum(prices, lookback=15)
    if not momentum or momentum["strength"] < 68 or momentum["consecutive"] < 5:
        return None
    direction = momentum["direction"]
    reasons = [f"{momentum['consecutive']} consecutive {direction} ticks"]
    candles = get_closed_candles(symbol)
    confidence = _apply_boosters(symbol, prices, direction, momentum["strength"] * 0.85, reasons, candles)
    signal = _build_signal(symbol, name, direction, "5s", confidence, prices[-1], " | ".join(reasons))
    if signal:
        try:
            save_pending_signal(signal["id"], symbol, "5s", direction, confidence,
                                {"rsi7": rsi(prices, 7) or 50, "rsi14": rsi(prices, 14) or 50})
        except:
            pass
    return signal

def signal_1min(symbol, name, prices):
    # Bitcoin and Gold excluded from 1min
    if symbol == "cryBTCUSD" or symbol == "frxXAUUSD":
        return None
    if not _session_allows_signal(symbol):
        return None
    if len(prices) < 70:
        return None
    sample = prices[-70:]
    ema5_now   = ema(sample, 5)
    ema13_now  = ema(sample, 13)
    ema5_prev  = ema(sample[:-5], 5)
    ema13_prev = ema(sample[:-5], 13)
    rsi_val    = rsi(sample, period=7)
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
    candles = get_closed_candles(symbol)
    confidence = _apply_boosters(symbol, prices, direction, base, reasons, candles)
    signal = _build_signal(symbol, name, direction, "1min", confidence, prices[-1], " | ".join(reasons))
    if signal:
        try:
            save_pending_signal(signal["id"], symbol, "1min", direction, confidence,
                                {"rsi7": rsi_val, "rsi14": rsi(sample, 14) or 50,
                                 "ema5": ema5_now, "ema13": ema13_now})
        except:
            pass
    return signal

def signal_3min(symbol, name, prices):
    # Bitcoin excluded from 3min
    if symbol == "cryBTCUSD":
        return None
    if not _session_allows_signal(symbol):
        return None
    if len(prices) < 160:
        return None
    sample = prices[-180:]
    rsi_val  = rsi(sample, period=14)
    macd_val = macd(sample, fast=8, slow=17, signal=9)
    stoch    = stochastic(sample, period=14)
    if not all([rsi_val, macd_val, stoch]):
        return None
    bull, bear, reasons = 0, 0, []
    if rsi_val < 35:   bull += 1; reasons.append(f"RSI oversold ({rsi_val})")
    elif rsi_val > 65: bear += 1; reasons.append(f"RSI overbought ({rsi_val})")
    if macd_val["cross"] == "bullish":   bull += 1; reasons.append("MACD bullish cross")
    elif macd_val["cross"] == "bearish": bear += 1; reasons.append("MACD bearish cross")
    elif macd_val["histogram"] > 0: bull += 1
    elif macd_val["histogram"] < 0: bear += 1
    if stoch["oversold"]:    bull += 1; reasons.append("Stochastic oversold")
    elif stoch["overbought"]: bear += 1; reasons.append("Stochastic overbought")
    for direction, score in [("BUY", bull), ("SELL", bear)]:
        if score >= 2:
            candles = get_closed_candles(symbol)
            confidence = _apply_boosters(symbol, prices, direction, 63 + score * 7, reasons, candles)
            signal = _build_signal(symbol, name, direction, "3min", confidence, prices[-1], " | ".join(reasons))
            if signal:
                try:
                    save_pending_signal(signal["id"], symbol, "3min", direction, confidence,
                                        {"rsi14": rsi_val, "macd_line": macd_val["macd_line"], "stoch_k": stoch["k"]})
                except:
                    pass
            return signal
    return None

def signal_5min(symbol, name, prices):
    if not _session_allows_signal(symbol):
        return None
    if len(prices) < 260:
        return None
    sample   = prices[-300:]
    rsi_val  = rsi(sample, period=14)
    macd_val = macd(sample, fast=12, slow=26, signal=9)
    bb       = bollinger_bands(sample, period=20, std_dev=2.0)
    if not all([rsi_val, macd_val, bb]):
        return None
    bull, bear, reasons = 0, 0, []
    if rsi_val < 35:   bull += 1; reasons.append(f"RSI oversold ({rsi_val})")
    elif rsi_val > 65: bear += 1; reasons.append(f"RSI overbought ({rsi_val})")
    if bb["percent_b"] < 0.25: bull += 1; reasons.append("Price near lower BB")
    elif bb["percent_b"] > 0.75: bear += 1; reasons.append("Price near upper BB")
    if macd_val["cross"] == "bullish":   bull += 2; reasons.append("MACD bullish cross")
    elif macd_val["histogram"] > 0:      bull += 1; reasons.append("MACD positive")
    elif macd_val["cross"] == "bearish": bear += 2; reasons.append("MACD bearish cross")
    elif macd_val["histogram"] < 0:      bear += 1; reasons.append("MACD negative")
    for direction, score in [("BUY", bull), ("SELL", bear)]:
        if score >= 3:
            candles = get_closed_candles(symbol)
            confidence = _apply_boosters(symbol, prices, direction, 68 + min(score * 5, 18), reasons, candles)
            signal = _build_signal(symbol, name, direction, "5min", confidence, prices[-1], " | ".join(reasons))
            if signal:
                try:
                    save_pending_signal(signal["id"], symbol, "5min", direction, confidence,
                                        {"rsi14": rsi_val, "macd_line": macd_val["macd_line"], "bb_percent_b": bb["percent_b"]})
                except:
                    pass
            return signal
    return None

STRATEGY_MAP = {
    "5s": signal_5s,
    "1min": signal_1min,
    "3min": signal_3min,
    "5min": signal_5min
}

def generate_signals():
    for symbol, name in get_all_symbols().items():
        prices = get_prices(symbol)
        if len(prices) < 60:
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
    if len(prices) < 60:
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
    session = is_forex_session()
    if session["dead_zone"]:
        return {
            "status": "low_liquidity",
            "message": "Markets in low liquidity hours (after 20:00 UTC / 10pm SAST). Come back during London session (7am UTC / 9am SAST).",
            "recommendations": []
        }
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
        sess = session_quality(sig["symbol"])
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
            "valid_for_seconds": sig.get("valid_for_seconds", 60),
            "valid_until": sig.get("valid_until"),
            "session_quality": sess["quality"],
            "session_note": sess["note"],
            "why": _explain(sig, sess),
            "reason": sig["reason"],
            "timestamp": sig["timestamp"],
        })
    return {
        "status": "ok",
        "session": session["session_label"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "recommendations": recommendations
    }

def _explain(sig, session):
    direction_text = "BUY — price going UP" if sig["direction"] == "BUY" else "SELL — price going DOWN"
    tf_map = {"5s": "5 seconds", "1min": "1 minute", "3min": "3 minutes", "5min": "5 minutes"}
    strength = "Very strong" if sig["confidence"] >= 85 else "Strong" if sig["confidence"] >= 75 else "Moderate"
    validity = sig.get("valid_for_seconds", 60)
    return (
        f"{strength} {direction_text} on {sig['asset']}. "
        f"Set expiry to {tf_map.get(sig['timeframe'], sig['timeframe'])}. "
        f"Confidence: {sig['confidence']}%. "
        f"Valid for {validity} seconds — act quickly. "
        f"Market: {session['note']}. "
        f"Indicators: {sig['reason']}."
    )
