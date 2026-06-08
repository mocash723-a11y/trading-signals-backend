"""
feedback.py — Trade Outcome Tracker + Persistent Storage
==========================================================
FIXES:
- Singleton MongoDB client (reused across calls)
- Load adaptive thresholds from MongoDB on startup
- TTL index for pending_signals (auto-delete old records)
"""

from datetime import datetime, timezone
from collections import defaultdict
import os
import atexit

# ── In-memory storage (always works) ─────────────────────────────────────────
outcomes: dict = defaultdict(lambda: defaultdict(list))
DEFAULT_THRESHOLD = 62
adaptive_thresholds: dict = defaultdict(lambda: defaultdict(lambda: DEFAULT_THRESHOLD))

# ── Singleton MongoDB client (FIX #1 & #11) ─────────────────────────────────
MONGO_URI = os.environ.get("MONGO_URI", "")
DB_NAME = "trading_signals"
COLL_PENDING = "pending_signals"
COLL_TRAINING = "training_examples"
COLL_THRESHOLDS = "adaptive_thresholds"

_mongo_client = None
_mongo_db = None
_mongo_available = False

def _get_mongo_client():
    """Singleton MongoDB client - created once and reused."""
    global _mongo_client, _mongo_db, _mongo_available
    if _mongo_client is not None:
        return _mongo_client, _mongo_db, _mongo_available
    
    if not MONGO_URI or MONGO_URI == "your_mongodb_atlas_uri":
        print("MongoDB: No URI provided, using memory-only mode")
        _mongo_available = False
        return None, None, False
    
    try:
        from pymongo import MongoClient
        _mongo_client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=5000,
            tls=True,
            tlsAllowInvalidCertificates=True
        )
        # Test connection once at startup
        _mongo_client.admin.command("ping")
        _mongo_db = _mongo_client[DB_NAME]
        _mongo_available = True
        print("MongoDB connected successfully (singleton client)")
        
        # Setup TTL index for pending_signals (FIX #12)
        try:
            _mongo_db[COLL_PENDING].create_index("timestamp", expireAfterSeconds=604800)  # 7 days
            print("TTL index on pending_signals created/verified")
        except Exception as e:
            print(f"TTL index warning: {e}")
        
        # Setup index for thresholds collection
        try:
            _mongo_db[COLL_THRESHOLDS].create_index([("symbol", 1), ("timeframe", 1)], unique=True)
        except Exception as e:
            pass
            
    except Exception as e:
        print(f"MongoDB not available: {e}")
        _mongo_available = False
        _mongo_client = None
        _mongo_db = None
    
    return _mongo_client, _mongo_db, _mongo_available

def _get_db():
    """Legacy wrapper - returns db or None."""
    _, db, available = _get_mongo_client()
    return db if available else None

def load_adaptive_thresholds_from_mongo():
    """Load saved thresholds from MongoDB on startup (FIX #3)."""
    _, db, available = _get_mongo_client()
    if not available or db is None:
        print("No MongoDB - using default thresholds only")
        return
    
    try:
        collection = db[COLL_THRESHOLDS]
        saved_thresholds = collection.find({})
        count = 0
        for doc in saved_thresholds:
            symbol = doc.get("symbol")
            timeframe = doc.get("timeframe")
            threshold = doc.get("threshold", DEFAULT_THRESHOLD)
            if symbol and timeframe:
                adaptive_thresholds[symbol][timeframe] = threshold
                count += 1
        print(f"Loaded {count} adaptive thresholds from MongoDB")
    except Exception as e:
        print(f"Failed to load thresholds: {e}")

def save_adaptive_threshold(symbol: str, timeframe: str, threshold: int):
    """Save a threshold to MongoDB."""
    _, db, available = _get_mongo_client()
    if not available or db is None:
        return
    
    try:
        collection = db[COLL_THRESHOLDS]
        collection.update_one(
            {"symbol": symbol, "timeframe": timeframe},
            {"$set": {"threshold": threshold, "updated_at": datetime.now(timezone.utc)}},
            upsert=True
        )
    except Exception as e:
        print(f"Failed to save threshold: {e}")

# ── Record trade outcome ──────────────────────────────────────────────────────

def record_outcome(symbol: str, timeframe: str, direction: str, outcome: str, confidence: int):
    entry = {
        "direction": direction,
        "outcome": outcome,
        "confidence": confidence,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    # Always save in memory
    outcomes[symbol][timeframe].append(entry)
    _update_threshold(symbol, timeframe)

    # Also save to MongoDB if available
    doc = {
        "symbol": symbol, "timeframe": timeframe,
        "direction": direction, "outcome": 1 if outcome == "win" else 0,
        "confidence": confidence,
        "timestamp": datetime.now(timezone.utc)
    }
    try:
        db = _get_db()
        if db is not None:
            db[COLL_TRAINING].insert_one(doc)
    except Exception as e:
        print(f"MongoDB write error (non-fatal): {e}")

    print(f"Feedback recorded: {symbol} {timeframe} {direction} → {outcome.upper()}")

def _update_threshold(symbol: str, timeframe: str):
    trades = outcomes[symbol][timeframe]
    if len(trades) < 5:
        return
    wins = sum(1 for t in trades if t["outcome"] == "win")
    win_rate = (wins / len(trades)) * 100
    current = adaptive_thresholds[symbol][timeframe]
    
    if win_rate < 45 and len(trades) >= 8:
        new_threshold = min(current + 5, 82)
        adaptive_thresholds[symbol][timeframe] = new_threshold
        save_adaptive_threshold(symbol, timeframe, new_threshold)  # Save to MongoDB
        print(f"Raising threshold {symbol} {timeframe}: {current}→{new_threshold} (wr {win_rate:.1f}%)")
    elif win_rate > 72 and len(trades) >= 8:
        new_threshold = max(current - 3, 55)
        adaptive_thresholds[symbol][timeframe] = new_threshold
        save_adaptive_threshold(symbol, timeframe, new_threshold)  # Save to MongoDB
        print(f"Lowering threshold {symbol} {timeframe}: {current}→{new_threshold} (wr {win_rate:.1f}%)")

def get_adaptive_threshold(symbol: str, timeframe: str) -> int:
    return adaptive_thresholds[symbol][timeframe]

def get_pair_accuracy(symbol: str, timeframe: str) -> dict:
    trades = outcomes[symbol][timeframe]
    if not trades:
        return {
            "symbol": symbol, "timeframe": timeframe,
            "total_trades": 0, "wins": 0, "losses": 0,
            "win_rate": None,
            "current_threshold": adaptive_thresholds[symbol][timeframe],
            "message": "No trades recorded yet"
        }
    wins = sum(1 for t in trades if t["outcome"] == "win")
    losses = len(trades) - wins
    win_rate = round((wins / len(trades)) * 100, 1)
    streak = _calculate_streak(trades)
    return {
        "symbol": symbol, "timeframe": timeframe,
        "total_trades": len(trades), "wins": wins, "losses": losses,
        "win_rate": win_rate,
        "current_threshold": adaptive_thresholds[symbol][timeframe],
        "current_streak": streak,
        "last_5": [t["outcome"] for t in trades[-5:]]
    }

def get_stats() -> dict:
    all_stats = []
    total_wins = total_losses = 0
    for symbol in outcomes:
        for timeframe in outcomes[symbol]:
            stats = get_pair_accuracy(symbol, timeframe)
            if stats["total_trades"] > 0:
                all_stats.append(stats)
                total_wins += stats["wins"]
                total_losses += stats["losses"]
    total_trades = total_wins + total_losses
    overall_win_rate = round((total_wins / total_trades) * 100, 1) if total_trades > 0 else None
    all_stats.sort(key=lambda x: x.get("win_rate") or 0, reverse=True)
    return {
        "overall": {
            "total_trades": total_trades, "wins": total_wins,
            "losses": total_losses, "win_rate": overall_win_rate
        },
        "by_pair": all_stats,
        "best_performer": all_stats[0] if all_stats else None,
        "worst_performer": all_stats[-1] if len(all_stats) > 1 else None,
        "mongo_connected": _mongo_available
    }

def _calculate_streak(trades: list) -> dict:
    if not trades:
        return {"type": "none", "count": 0}
    streak_type = trades[-1]["outcome"]
    count = 0
    for trade in reversed(trades):
        if trade["outcome"] == streak_type:
            count += 1
        else:
            break
    return {"type": streak_type, "count": count}

def save_pending_signal(signal_id, symbol, timeframe, direction, confidence, indicators):
    """Save signal with its indicator state before outcome is known — for ML training."""
    doc = {
        "signal_id": signal_id, "symbol": symbol,
        "timeframe": timeframe, "direction": direction,
        "confidence": confidence, "outcome": "pending",
        "features": indicators,
        "timestamp": datetime.now(timezone.utc)
    }
    try:
        db = _get_db()
        if db is not None:
            db[COLL_PENDING].insert_one(doc)
            # No longer need to print on every save
    except Exception as e:
        print(f"Pending signal save error (non-fatal): {e}")

def update_signal_outcome(signal_id, outcome):
    """Link a WIN/LOSS result to the specific signal that generated it."""
    try:
        db = _get_db()
        if db is not None:
            result = db[COLL_PENDING].update_one(
                {"signal_id": signal_id},
                {"$set": {"outcome": outcome, "resolved_at": datetime.now(timezone.utc)}}
            )
            if result.matched_count == 0:
                print(f"Signal {signal_id} not found (already resolved or never saved)")
    except Exception as e:
        print(f"Update outcome error (non-fatal): {e}")
