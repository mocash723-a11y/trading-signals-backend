# train_model.py - Trains separate AI models for BUY and SELL

import os
import json
import joblib
import numpy as np
from datetime import datetime
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

print(f"🤖 AI Training Started at {datetime.utcnow().isoformat()}")

MONGO_URI = os.environ.get("MONGO_URI")
if not MONGO_URI:
    print("❌ MONGO_URI not found")
    exit(1)

client = MongoClient(MONGO_URI)
db = client["trading_signals"]

# Get all trades with outcome win/loss and non‑empty features
trades = list(db.training_data.find({
    "outcome": {"$in": ["win", "loss"]},
    "features": {"$ne": {}}
}))
print(f"📊 Found {len(trades)} trades with indicators")

if len(trades) < 50:
    print(f"⚠️ Need at least 50 trades. Have {len(trades)}. Skipping training.")
    exit(0)

# Split into BUY and SELL trades
buy_trades = [t for t in trades if t.get("direction") == "BUY"]
sell_trades = [t for t in trades if t.get("direction") == "SELL"]
print(f"   BUY trades: {len(buy_trades)}, SELL trades: {len(sell_trades)}")

def prepare_features(trade_list):
    X, y = [], []
    for trade in trade_list:
        f = trade["features"]
        # Features (order must match ml_confidence in signals.py)
        X.append([
            f.get("rsi7", 50),
            f.get("rsi14", 50),
            f.get("macd_line", 0),
            f.get("stoch_k", 50),
            f.get("bb_percent_b", 0.5),
            f.get("adx", 25),                # new
            f.get("atr", 0.001),             # new
            f.get("session_multiplier", 1.0),# new
            1 if trade["direction"] == "BUY" else 0   # direction (only needed for combined, but we split)
        ])
        y.append(1 if trade["outcome"] == "win" else 0)
    return X, y

# Train BUY model if enough data
if len(buy_trades) >= 30:
    X_buy, y_buy = prepare_features(buy_trades)
    model_buy = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model_buy.fit(X_buy, y_buy)
    joblib.dump(model_buy, "model_buy.pkl")
    buy_win_rate = sum(y_buy)/len(y_buy)*100
    print(f"✅ BUY model trained on {len(buy_trades)} trades (win rate: {buy_win_rate:.1f}%)")
else:
    print(f"⚠️ Not enough BUY trades ({len(buy_trades)}), skipping BUY model")

# Train SELL model if enough data
if len(sell_trades) >= 30:
    X_sell, y_sell = prepare_features(sell_trades)
    model_sell = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model_sell.fit(X_sell, y_sell)
    joblib.dump(model_sell, "model_sell.pkl")
    sell_win_rate = sum(y_sell)/len(y_sell)*100
    print(f"✅ SELL model trained on {len(sell_trades)} trades (win rate: {sell_win_rate:.1f}%)")
else:
    print(f"⚠️ Not enough SELL trades ({len(sell_trades)}), skipping SELL model")

# Save metadata (combined)
metadata = {
    "last_trained": datetime.utcnow().isoformat(),
    "total_trades": len(trades),
    "buy_trades": len(buy_trades),
    "sell_trades": len(sell_trades),
    "overall_win_rate": sum(1 for t in trades if t["outcome"]=="win")/len(trades)*100,
}
if len(buy_trades) >= 30:
    metadata["buy_model_win_rate"] = buy_win_rate
if len(sell_trades) >= 30:
    metadata["sell_model_win_rate"] = sell_win_rate

with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("🎉 Training complete.")
