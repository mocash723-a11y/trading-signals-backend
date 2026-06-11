# train_model.py - AI Training Script
# Runs automatically via Render cron job or manually via /train-ai endpoint

import os
import json
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
    print("❌ MONGO_URI not found! Set environment variable.")
    exit(1)

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client["trading_signals"]
    # Test connection
    client.admin.command('ping')
    print("✅ Connected to MongoDB")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    exit(1)

# Get ALL completed trades that have indicator data (features)
# This includes:
#   - Trades saved from Recommendations tab (with features & outcome)
#   - Trades from Signal Generator (with features & outcome after feedback)
signals = list(db.pending_signals.find({
    "outcome": {"$in": ["win", "loss"]},
    "features": {"$ne": {}}
}))

print(f"📊 Found {len(signals)} trades with indicators AND outcomes")

if len(signals) < 50:
    print(f"⚠️ Need at least 50 trades. Currently have {len(signals)}. Skipping training.")
    print("   Keep trading and submitting feedback. The AI will train automatically once enough data exists.")
    exit(0)

# Prepare feature vectors and labels
X, y = [], []
for sig in signals:
    f = sig.get("features", {})
    # Order must match what ml_confidence() in signals.py expects
    X.append([
        f.get("rsi7", 50),           # RSI 7
        f.get("rsi14", 50),          # RSI 14
        f.get("macd_line", 0),       # MACD line
        f.get("stoch_k", 50),        # Stochastic %K
        f.get("bb_percent_b", 0.5),  # Bollinger %B
        1 if sig.get("direction") == "BUY" else 0  # Direction (1=BUY, 0=SELL)
    ])
    y.append(1 if sig["outcome"] == "win" else 0)

X = np.array(X)
y = np.array(y)

print(f"📈 Training data shape: {X.shape}")
print(f"🏆 Win rate in training data: {sum(y)/len(y)*100:.1f}%")

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)

# Cross-validation accuracy
scores = cross_val_score(model, X, y, cv=5)
print(f"🎯 Model cross-validation accuracy: {scores.mean()*100:.1f}% (±{scores.std()*100:.1f}%)")

# Feature importance
feature_names = ["RSI7", "RSI14", "MACD", "StochK", "BB%B", "Direction"]
importance = dict(zip(feature_names, model.feature_importances_))
print(f"📊 Feature importance: {importance}")

# Save the model
joblib.dump(model, "model.pkl")
print("✅ Model saved to model.pkl")

# Save training metadata
metadata = {
    "last_trained": datetime.utcnow().isoformat(),
    "total_trades": len(signals),
    "wins": int(sum(y)),
    "losses": int(len(y) - sum(y)),
    "win_rate": round(sum(y)/len(y)*100, 1),
    "cv_accuracy": round(scores.mean()*100, 1),
    "feature_importance": {k: round(v, 3) for k, v in importance.items()}
}
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("📋 Training metadata saved to model_metadata.json")
print("🎉 AI Training Complete!")
