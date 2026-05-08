from datetime import datetime, timezone
from collections import defaultdict
from pymongo import MongoClient
import os

# ── In-memory storage ─────────────────────────────────────────────────────────
outcomes: dict = defaultdict(lambda: defaultdict(list))
DEFAULT_THRESHOLD = 62
adaptive_thresholds: dict = defaultdict(lambda: defaultdict(lambda: DEFAULT_THRESHOLD))

# ── MongoDB configuration ─────────────────────────────────────────────────────
MONGO_URI = os.environ.get("MONGO_URI", "your_mongodb_atlas_uri")
DB_NAME = "trading_signals"
COLL_PENDING = "pending_signals"
COLL_TRAINING = "training_examples"

def get_mongo_client():
    return MongoClient(MONGO_URI)

# ── Record a trade outcome ────────────────────────────────────────────────────
def record_outcome(symbol: str, timeframe: str, direction: str, outcome: str, confidence: int):
    entry = {
        "direction": direction,
        "outcome": outcome,
        "confidence": confidence,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    outcomes[symbol][timeframe].append(entry)
    _update_threshold(symbol, timeframe)
    _save_training_example(symbol, timeframe, direction, outcome, confidence)
    print(f"Feedback recorded: {symbol} {timeframe} {direction} → {outcome.upper()}")

def _update_threshold(symbol: str, timeframe: str):
    trades = outcomes[symbol][timeframe]
    if len(trades) < 5:
        return
    wins = sum(1 for t in trades if t["outcome"] == "win")
    win_rate = (wins / len(trades)) * 100
    current = adaptive_thresholds[symbol][timeframe]
    if win_rate < 45 and len(trades) >= 8:
        adaptive_thresholds[symbol][timeframe] = min(current + 5, 82)
        print(f"Raising threshold {symbol} {timeframe}: {current} → {adaptive_thresholds[symbol][timeframe]} (wr {win_rate:.1f}%)")
    elif win_rate > 72 and len(trades) >= 8:
        adaptive_thresholds[symbol][timeframe] = max(current - 3, 55)
        print(f"Lowering threshold {symbol} {timeframe}: {current} → {adaptive_thresholds[symbol][timeframe]} (wr {win_rate:.1f}%)")

def get_adaptive_threshold(symbol: str, timeframe: str) -> int:
    return adaptive_thresholds[symbol][timeframe]

def get_pair_accuracy(symbol: str, timeframe: str) -> dict:
    trades = outcomes[symbol][timeframe]
    if not trades:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": None,
            "current_threshold": adaptive_thresholds[symbol][timeframe],
            "message": "No trades recorded yet"
        }
    wins = sum(1 for t in trades if t["outcome"] == "win")
    losses = len(trades) - wins
    win_rate = round((wins / len(trades)) * 100, 1)
    streak = _calculate_streak(trades)
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "total_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "current_threshold": adaptive_thresholds[symbol][timeframe],
        "current_streak": streak,
        "last_5": [t["outcome"] for t in trades[-5:]]
    }

def get_stats() -> dict:
    all_stats = []
    total_wins = 0
    total_losses = 0
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
            "total_trades": total_trades,
            "wins": total_wins,
            "losses": total_losses,
            "win_rate": overall_win_rate
        },
        "by_pair": all_stats,
        "best_performer": all_stats[0] if all_stats else None,
        "worst_performer": all_stats[-1] if len(all_stats) > 1 else None
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

# ── Save training example (old file method kept for backward compat) ──────────
def _save_training_example(symbol, timeframe, direction, outcome, confidence):
    # This now stores in MongoDB Atlas for persistence
    doc = {
        "symbol": symbol,
        "timeframe": timeframe,
        "direction": direction,
        "outcome": 1 if outcome == "win" else 0,
        "confidence": confidence,
        "timestamp": datetime.now(timezone.utc)
    }
    try:
        client = get_mongo_client()
        db = client[DB_NAME]
        db[COLL_TRAINING].insert_one(doc)
        client.close()
    except Exception as e:
        print(f"MongoDB write error: {e}")

# ── New functions for pending signals (ML feature capture) ────────────────────
def save_pending_signal(signal_id, symbol, timeframe, direction, confidence, indicators):
    doc = {
        "signal_id": signal_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "direction": direction,
        "confidence": confidence,
        "timestamp": datetime.now(timezone.utc),
        "outcome": "pending",
        "features": indicators
    }
    try:
        client = get_mongo_client()
        db = client[DB_NAME]
        db[COLL_PENDING].insert_one(doc)
        client.close()
    except Exception as e:
        print(f"Pending signal save error: {e}")

def update_signal_outcome(signal_id, outcome):
    try:
        client = get_mongo_client()
        db = client[DB_NAME]
        db[COLL_PENDING].update_one(
            {"signal_id": signal_id},
            {"$set": {"outcome": outcome, "resolved_at": datetime.now(timezone.utc)}}
        )
        client.close()
        print(f"Updated pending signal {signal_id} to {outcome}")
    except Exception as e:
        print(f"Update outcome error: {e}")
