# train_model.py - Runs automatically on Render every week
import os
import joblib
import numpy as np
from datetime import datetime
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

print(f"🤖 AI Training Started at {datetime.utcnow().isoformat()}")

# Connect to MongoDB
MONGO_URI = os.environ.get("MONGO_URI")
if not MONGO_URI:
    print("❌ MONGO_URI not found!")
    exit(1)

client = MongoClient(MONGO_URI)
db = client["trading_signals"]

# Get ALL completed trades with indicator data
signals = list(db.pending_signals.find({
    "outcome": {"$in": ["win", "loss"]},
    "features": {"$ne": {}}  # Must have indicator data
}))

print(f"📊 Found {len(signals)} completed trades with indicator data")

if len(signals) < 50:
    print(f"⚠️ Need at least 50 trades. Currently have {len(signals)}. Skipping training.")
    exit(0)

# Prepare features and labels
X, y = [], []
for sig in signals:
    f = sig.get("features", {})
    X.append([
        f.get("rsi7", 50),           # RSI 7 period
        f.get("rsi14", 50),          # RSI 14 period  
        f.get("macd_line", 0),       # MACD line
        f.get("stoch_k", 50),        # Stochastic %K
        f.get("bb_percent_b", 0.5),  # Bollinger Band %B
        1 if sig.get("direction") == "BUY" else 0  # Direction
    ])
    y.append(1 if sig["outcome"] == "win" else 0)

X = np.array(X)
y = np.array(y)

print(f"📈 Training data shape: {X.shape}")

# Train the model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
model.fit(X, y)

# Cross-validation score
scores = cross_val_score(model, X, y, cv=5)
print(f"🎯 Model accuracy: {scores.mean()*100:.1f}% (±{scores.std()*100:.1f}%)")

# Calculate feature importance
feature_names = ["RSI7", "RSI14", "MACD", "StochK", "BB%B", "Direction"]
importance = dict(zip(feature_names, model.feature_importances_))
print(f"📊 Feature importance: {importance}")

# Save the model
joblib.dump(model, "model.pkl")
print(f"✅ Model saved to model.pkl")

# Also save training metadata
metadata = {
    "last_trained": datetime.utcnow().isoformat(),
    "total_trades": len(signals),
    "wins": sum(y),
    "losses": len(y) - sum(y),
    "win_rate": sum(y)/len(y)*100,
    "accuracy": scores.mean()*100,
    "feature_importance": importance
}

import json
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"📋 Training metadata saved")
print(f"🏆 Win rate in training data: {sum(y)/len(y)*100:.1f}%")
