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
    adx, atr, detect_candle_patterns_from_ohlc
)
from feedback import get_adaptive_threshold, save_pending_signal

import joblib
import os

# Load separate models for BUY and SELL
ml_model_buy = None
ml_model_sell = None
if os.path.exists("model_buy.pkl"):
    try:
        ml_model_buy = joblib.load("model_buy.pkl")
        print("AI model (BUY) loaded successfully.")
    except Exception as e:
        print(f"Model load error (BUY): {e}")
if os.path.exists("model_sell.pkl"):
    try:
        ml_model_sell = joblib.load("model_sell.pkl")
        print("AI model (SELL) loaded successfully.")
    except Exception as e:
        print(f"Model load error (SELL): {e}")

if not ml_model_buy and not ml_model_sell:
    print("No AI models found — using rule-based confidence only.")

current_signals = {}

def _is_signal_fresh(symbol, timeframe):
    cache_key = f"{symbol}_{timeframe}"
    if cache_key not in current_signals:
        return False
    existing = current_signals[cache_key]
    valid_until = existing.get("valid_until")
    if not valid_until:
        return False
    try:
        valid_until_dt = datetime.fromisoformat(valid_until.replace('Z', '+00:00'))
        return datetime.now(timezone.utc) < valid_until_dt
    except:
        return False

def _get_min_confidence(symbol, timeframe):
    pair_default = get_pair_min_confidence(symbol)
    adaptive = get_adaptive_threshold(symbol, timeframe)
    return max(pair_default, adaptive)

def _build_signal(symbol, name, direction, timeframe, confidence, entry_price, reason, extras=None):
    confidence = float(confidence)
    confidence = min(confidence, 100.0)  # cap at 100%
    confidence_int = int(round(confidence))
    min_conf = _get_min_confidence(symbol, timeframe)
    if confidence_int < min_conf:
        return None
    validity_seconds = get_signal_validity(timeframe)
    signal = {
        "id": f"{symbol}_{timeframe}_{int(datetime.now().timestamp())}",
        "asset": name,
        "symbol": symbol,
        "direction": direction,
        "timeframe": timeframe,
        "entry_price": round(entry_price, 5),
        "confidence": confidence_int,
        "is_vip": bool(confidence >= 78),
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
    confidence = float(base_confidence)
    
    # Session multiplier (applied FIRST)
    session = session_quality(symbol)
    confidence *= session["multiplier"]
    if session["quality"] == "excellent":
        reasons.append("Peak session")
    elif session["quality"] == "good":
        reasons.append("Active session")
    elif session["quality"] == "poor":
        reasons.append("Low liquidity")
    
    # Multi-timeframe confirmation (reduced bonus)
    mtf = multi_timeframe_confirm(symbol, direction)
    if mtf["confirmed"]:
        confidence += mtf["bonus"]
        reasons.append(f"Multi-TF confirmed ({mtf['agreement']} agree)")
    
    # Candle pattern bonus (reduced)
    if candles_1m and len(candles_1m) >= 2:
        candle = detect_candle_patterns_from_ohlc(candles_1m)
    else:
        candle = detect_candle_patterns(prices)
    if candle and candle["bias"] == direction and candle["bonus"] > 0:
        confidence += candle["bonus"]
        reasons.append(f"{candle['pattern']} detected")
    
    # ADX penalty and logging
    if not symbol.startswith("cry") and candles_1m and len(candles_1m) >= 15:
        highs = [c["high"] for c in candles_1m[-15:]]
        lows = [c["low"] for c in candles_1m[-15:]]
        closes = [c["close"] for c in candles_1m[-15:]]
        adx_val = adx(highs, lows, closes, period=14)
        if adx_val:
            adx_value = adx_val["adx"]
            if adx_value < 20:
                confidence *= 0.85
                reasons.append("Low ADX — ranging market (confidence reduced)")
                print(f"📉 ADX Log [{symbol}]: {adx_value:.2f} → LOW trend (penalty applied)")
            else:
                print(f"📈 ADX Log [{symbol}]: {adx_value:.2f} → GOOD trend (no penalty)")
        else:
            print(f"⚠️ ADX Log [{symbol}]: ADX calculation returned None")
    
    # Final cap at 100%
    confidence = min(confidence, 100.0)
    return confidence

def _session_allows_signal(symbol):
    weekday = datetime.utcnow().weekday()
    if symbol.startswith("cry"):
        return True
    if weekday >= 5:
        return False
    return True

def _get_real_time_features(symbol, direction):
    """Compute features for the ML model using real data (candles + prices)."""
    try:
        candles = get_closed_candles(symbol)
        if not candles or len(candles) < 20:
            return None
        highs = [c["high"] for c in candles[-20:]]
        lows = [c["low"] for c in candles[-20:]]
        closes = [c["close"] for c in candles[-20:]]
        adx_val = adx(highs, lows, closes, period=14)
        atr_val = atr(highs, lows, closes, period=14)
        
        prices = get_prices(symbol)
        if len(prices) < 30:
            return None
        
        features = [
            rsi(prices, 7) or 50,
            rsi(prices, 14) or 50,
            macd(prices, 12, 26, 9)["macd_line"] if macd(prices, 12, 26, 9) else 0,
            stochastic(prices, 14)["k"] if stochastic(prices, 14) else 50,
            bollinger_bands(prices, 20, 2.0)["percent_b"] if bollinger_bands(prices, 20, 2.0) else 0.5,
            adx_val["adx"] if adx_val else 25,
            atr_val if atr_val else 0.001,
            session_quality(symbol)["multiplier"],
            1 if direction == "BUY" else 0
        ]
        return features
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

def ml_confidence(symbol, timeframe, direction, prices):
    """Use separate models for BUY and SELL."""
    model = ml_model_buy if direction == "BUY" else ml_model_sell
    if not model:
        return None
    try:
        features = _get_real_time_features(symbol, direction)
        if not features:
            return None
        prob = model.predict_proba([features])[0][1]
        confidence = prob * 100
        return round(confidence, 1)
    except Exception as e:
        print(f"ML confidence error: {e}")
        return None

# ------------------------- 5s signal -------------------------
def signal_5s(symbol, name, prices):
    if _is_signal_fresh(symbol, "5s"):
        return current_signals.get(f"{symbol}_5s")
    if symbol == "cryBTCUSD" or symbol in GOLD_EXCLUDED_TIMEFRAMES or symbol == "frxXAUUSD":
        return None
    if not _session_allows_signal(symbol):
        return None
    if len(prices) < 20:
        return None
    momentum = tick_momentum(prices, lookback=15)
    if not momentum or momentum["strength"] < 68 or momentum["consecutive"] < 5:
        return None
    raw_direction = momentum["direction"]
    direction = "BUY" if raw_direction == "UP" else "SELL"
    reasons = [f"{momentum['consecutive']} consecutive {raw_direction} ticks"]
    base_conf = momentum["strength"] * 0.85
    ml_conf = ml_confidence(symbol, "5s", direction, prices)
    base_conf = ml_conf if ml_conf else base_conf
    candles = get_closed_candles(symbol)
    confidence = _apply_boosters(symbol, prices, direction, base_conf, reasons, candles)
    indicators = {
        "rsi7": rsi(prices, 7) or 50,
        "rsi14": rsi(prices, 14) or 50,
        "macd_line": macd(prices, 12, 26, 9)["macd_line"] if macd(prices, 12, 26, 9) else 0,
        "stoch_k": stochastic(prices, 14)["k"] if stochastic(prices, 14) else 50,
        "bb_percent_b": bollinger_bands(prices, 20, 2.0)["percent_b"] if bollinger_bands(prices, 20, 2.0) else 0.5,
        "direction_encoded": 1 if direction == "BUY" else 0
    }
    # Add ADX/ATR/session if candles available
    if candles and len(candles) >= 15:
        highs = [c["high"] for c in candles[-15:]]
        lows = [c["low"] for c in candles[-15:]]
        closes = [c["close"] for c in candles[-15:]]
        adx_val = adx(highs, lows, closes, period=14)
        atr_val = atr(highs, lows, closes, period=14)
        if adx_val:
            indicators["adx"] = adx_val["adx"]
        if atr_val:
            indicators["atr"] = atr_val
        indicators["session_multiplier"] = session_quality(symbol)["multiplier"]
    signal = _build_signal(symbol, name, direction, "5s", confidence, prices[-1], " | ".join(reasons), extras={"indicators": indicators})
    if signal:
        try:
            save_pending_signal(signal["id"], symbol, "5s", direction, confidence, indicators)
        except Exception as e:
            print(f"Failed to save pending signal for {symbol} 5s: {e}")
    return signal

# ------------------------- 1min signal -------------------------
def signal_1min(symbol, name, prices):
    if _is_signal_fresh(symbol, "1min"):
        return current_signals.get(f"{symbol}_1min")
    if symbol == "cryBTCUSD" or symbol == "frxXAUUSD":
        return None
    if not _session_allows_signal(symbol):
        return None
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
    ml_conf = ml_confidence(symbol, "1min", direction, prices)
    base = ml_conf if ml_conf else base
    candles = get_closed_candles(symbol)
    confidence = _apply_boosters(symbol, prices, direction, base, reasons, candles)
    indicators = {
        "rsi7": rsi_val,
        "rsi14": rsi(sample, 14) or 50,
        "macd_line": macd(sample, 12, 26, 9)["macd_line"] if macd(sample, 12, 26, 9) else 0,
        "stoch_k": stochastic(sample, 14)["k"] if stochastic(sample, 14) else 50,
        "bb_percent_b": bollinger_bands(sample, 20, 2.0)["percent_b"] if bollinger_bands(sample, 20, 2.0) else 0.5,
        "direction_encoded": 1 if direction == "BUY" else 0
    }
    if candles and len(candles) >= 15:
        highs = [c["high"] for c in candles[-15:]]
        lows = [c["low"] for c in candles[-15:]]
        closes = [c["close"] for c in candles[-15:]]
        adx_val = adx(highs, lows, closes, period=14)
        atr_val = atr(highs, lows, closes, period=14)
        if adx_val:
            indicators["adx"] = adx_val["adx"]
        if atr_val:
            indicators["atr"] = atr_val
        indicators["session_multiplier"] = session_quality(symbol)["multiplier"]
    signal = _build_signal(symbol, name, direction, "1min", confidence, prices[-1], " | ".join(reasons), extras={"indicators": indicators})
    if signal:
        try:
            save_pending_signal(signal["id"], symbol, "1min", direction, confidence, indicators)
        except Exception as e:
            print(f"Failed to save pending signal for {symbol} 1min: {e}")
    return signal

# ------------------------- 3min signal -------------------------
def signal_3min(symbol, name, prices):
    if _is_signal_fresh(symbol, "3min"):
        return current_signals.get(f"{symbol}_3min")
    if symbol == "cryBTCUSD":
        return None
    if not _session_allows_signal(symbol):
        return None
    if len(prices) < 160:
        return None
    sample = prices[-180:]
    rsi_val = rsi(sample, period=14)
    macd_val = macd(sample, fast=8, slow=17, signal=9)
    stoch = stochastic(sample, period=14)
    if not all([rsi_val, macd_val, stoch]):
        return None
    bull, bear, reasons = 0, 0, []
    if rsi_val < 35:
        bull += 1
        reasons.append(f"RSI oversold ({rsi_val})")
    elif rsi_val > 65:
        bear += 1
        reasons.append(f"RSI overbought ({rsi_val})")
    if macd_val["cross"] == "bullish":
        bull += 1
        reasons.append("MACD bullish cross")
    elif macd_val["cross"] == "bearish":
        bear += 1
        reasons.append("MACD bearish cross")
    elif macd_val["histogram"] > 0:
        bull += 1
    elif macd_val["histogram"] < 0:
        bear += 1
    if stoch["oversold"]:
        bull += 1
        reasons.append("Stochastic oversold")
    elif stoch["overbought"]:
        bear += 1
        reasons.append("Stochastic overbought")
    for direction, score in [("BUY", bull), ("SELL", bear)]:
        if score >= 2:
            base = 63 + score * 7
            ml_conf = ml_confidence(symbol, "3min", direction, prices)
            base = ml_conf if ml_conf else base
            candles = get_closed_candles(symbol)
            confidence = _apply_boosters(symbol, prices, direction, base, reasons, candles)
            indicators = {
                "rsi7": rsi(sample, 7) or 50,
                "rsi14": rsi_val,
                "macd_line": macd_val["macd_line"],
                "stoch_k": stoch["k"],
                "bb_percent_b": bollinger_bands(sample
