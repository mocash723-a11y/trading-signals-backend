"""
feedback.py — Trade Outcome Tracker & Adaptive Learning
=========================================================
Stores WIN/LOSS results submitted by the user after each trade.
Uses this data to:
1. Track your personal win rate per pair and timeframe
2. Automatically raise the confidence threshold for pairs that
   are performing poorly (fewer bad signals shown)
3. Reward pairs with high win rates by showing them more prominently

Data is stored in memory (resets when server restarts on Render).
The Lovable frontend also stores results in localStorage as a
permanent backup on your device.

How the adaptive system works:
- If a pair/timeframe has 10+ trades and win rate < 50% → raise threshold by 5
- If a pair/timeframe has 10+ trades and win rate > 70% → lower threshold by 3
  (shows more signals from reliable pairs)
- This self-corrects over time based purely on your real trade results
"""

from datetime import datetime, timezone
from collections import defaultdict

# ── In-memory storage ─────────────────────────────────────────────────────────
# Structure: outcomes[symbol][timeframe] = list of result dicts
outcomes: dict = defaultdict(lambda: defaultdict(list))

# Adaptive thresholds — start at default and adjust based on feedback
# Structure: thresholds[symbol][timeframe] = min_confidence int
DEFAULT_THRESHOLD = 62
adaptive_thresholds: dict = defaultdict(lambda: defaultdict(lambda: DEFAULT_THRESHOLD))

# ── Record a trade outcome ────────────────────────────────────────────────────

def record_outcome(symbol: str, timeframe: str, direction: str, outcome: str, confidence: int):
    """
    Store a WIN or LOSS result and update the adaptive threshold.
    Called by the /feedback endpoint.
    """
    entry = {
        "direction": direction,
        "outcome": outcome,          # "win" or "loss"
        "confidence": confidence,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    outcomes[symbol][timeframe].append(entry)

    # Recalculate adaptive threshold after every trade
    _update_threshold(symbol, timeframe)

    print(f"Feedback recorded: {symbol} {timeframe} {direction} → {outcome.upper()}")

def _update_threshold(symbol: str, timeframe: str):
    """
    Adjust the minimum confidence threshold for a pair/timeframe
    based on its historical win rate.
    Only kicks in after 5+ trades (needs data to be meaningful).
    """
    trades = outcomes[symbol][timeframe]
    if len(trades) < 5:
        return

    wins = sum(1 for t in trades if t["outcome"] == "win")
    win_rate = (wins / len(trades)) * 100

    current = adaptive_thresholds[symbol][timeframe]

    if win_rate < 45 and len(trades) >= 8:
        # Performing poorly — raise threshold, show fewer but better signals
        adaptive_thresholds[symbol][timeframe] = min(current + 5, 82)
        print(f"Raising threshold for {symbol} {timeframe}: {current} → {adaptive_thresholds[symbol][timeframe]} (win rate {win_rate:.1f}%)")

    elif win_rate > 72 and len(trades) >= 8:
        # Performing well — lower threshold slightly, show more signals
        adaptive_thresholds[symbol][timeframe] = max(current - 3, 55)
        print(f"Lowering threshold for {symbol} {timeframe}: {current} → {adaptive_thresholds[symbol][timeframe]} (win rate {win_rate:.1f}%)")

# ── Public accessors ──────────────────────────────────────────────────────────

def get_adaptive_threshold(symbol: str, timeframe: str) -> int:
    """
    Used by signals.py to get the current minimum confidence for a pair.
    Starts at 62, adjusts based on feedback data.
    """
    return adaptive_thresholds[symbol][timeframe]

def get_pair_accuracy(symbol: str, timeframe: str) -> dict:
    """Returns win/loss stats for a specific pair and timeframe."""
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
    """Returns full stats across all pairs and timeframes."""
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

    # Sort by win rate descending
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
    """Calculate current winning or losing streak."""
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
