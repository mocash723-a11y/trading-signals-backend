# train_model.py - Trains AI from permanent training_data collection

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


trades = list(db.training_data.find({"outcome": {"$in": ["win", "loss"]}, "features": {"$ne": {}}}))


print(f"📊 Found {len(trades)} trades in training_data with indicators")

if len(trades) < 50:
    print(f"⚠️ Need at least 50 trades. Have {len(trades)}. Skipping training.")
    exit(0)

X, y = [], []
for trade in trades:
    f = trade["features"]
    X.append([
        f.get("rsi7", 50),
        f.get("rsi14", 50),
        f.get("macd_line", 0),
        f.get("stoch_k", 50),
        f.get("bb_percent_b", 0.5),
        1 if trade["direction"] == "BUY" else 0
    ])
    y.append(1 if trade["outcome"] == "win" else 0)

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X, y)

joblib.dump(model, "model.pkl")

# Save metadata
wins = sum(y)
metadata = {
    "last_trained": datetime.utcnow().isoformat(),
    "total_trades": len(trades),
    "wins": wins,
    "losses": len(trades) - wins,
    "win_rate": round(wins/len(trades)*100, 1),
    "feature_importance": dict(zip(["rsi7","rsi14","macd","stochK","bb%b","dir"], model.feature_importances_))
}
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f)

print(f"✅ Model trained on {len(trades)} trades. Win rate: {metadata['win_rate']}%")
