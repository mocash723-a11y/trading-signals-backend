# train_model.py
import os
import joblib
import numpy as np
from datetime import datetime
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier

MONGO_URI = os.environ.get("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["trading_signals"]

# Get completed trades with features
signals = list(db.pending_signals.find({
    "outcome": {"$in": ["win", "loss"]},
    "features": {"$ne": {}}
}))

print(f"Found {len(signals)} trades with indicators and outcomes")

if len(signals) < 50:
    print("Not enough data. Need at least 50.")
    exit(0)

X, y = [], []
for s in signals:
    f = s["features"]
    X.append([
        f.get("rsi7", 50),
        f.get("rsi14", 50),
        f.get("macd_line", 0),
        f.get("stoch_k", 50),
        f.get("bb_percent_b", 0.5),
        1 if s.get("direction") == "BUY" else 0
    ])
    y.append(1 if s["outcome"] == "win" else 0)

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

joblib.dump(model, "model.pkl")

# Save metadata
import json
metadata = {
    "last_trained": datetime.utcnow().isoformat(),
    "total_trades": len(signals),
    "wins": sum(y),
    "losses": len(y)-sum(y),
    "win_rate": sum(y)/len(y)*100,
    "feature_importance": dict(zip(["rsi7","rsi14","macd","stochK","bb%b","dir"], model.feature_importances_))
}
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f)

print(f"Model trained on {len(signals)} trades. Win rate: {sum(y)/len(y)*100:.1f}%")
