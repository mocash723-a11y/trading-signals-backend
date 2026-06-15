# train_model.py - Trains separate models with feature importance

import os
import json
import joblib
import numpy as np
from datetime import datetime
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier

print(f"🤖 AI Training Started at {datetime.utcnow().isoformat()}")

MONGO_URI = os.environ.get("MONGO_URI")
if not MONGO_URI:
    print("❌ MONGO_URI not found")
    exit(1)

client = MongoClient(MONGO_URI)
db = client["trading_signals"]

trades = list(db.training_data.find({
    "outcome": {"$in": ["win", "loss"]},
    "features": {"$ne": {}}
}))
print(f"📊 Found {len(trades)} trades with indicators")

if len(trades) < 50:
    print(f"⚠️ Need at least 50 trades. Have {len(trades)}. Skipping training.")
    exit(0)

# Split into BUY and SELL
buy_trades = [t for t in trades if t.get("direction") == "BUY"]
sell_trades = [t for t in trades if t.get("direction") == "SELL"]
print(f"   BUY: {len(buy_trades)}, SELL: {len(sell_trades)}")

feature_names = ["rsi7", "rsi14", "macd", "stochK", "bb%b", "adx", "atr", "session", "dir"]

def extract_features(trade_list):
    X, y = [], []
    for trade in trade_list:
        f = trade["features"]
        X.append([
            f.get("rsi7", 50),
            f.get("rsi14", 50),
            f.get("macd_line", 0),
            f.get("stoch_k", 50),
            f.get("bb_percent_b", 0.5),
            f.get("adx", 25),
            f.get("atr", 0.001),
            f.get("session_multiplier", 1.0),
            1 if trade["direction"] == "BUY" else 0
        ])
        y.append(1 if trade["outcome"] == "win" else 0)
    return X, y

# Train SELL model (usually has more data)
sell_model = None
sell_importance = None
if len(sell_trades) >= 30:
    X_sell, y_sell = extract_features(sell_trades)
    sell_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    sell_model.fit(X_sell, y_sell)
    joblib.dump(sell_model, "model_sell.pkl")
    sell_win_rate = sum(y_sell)/len(y_sell)*100
    sell_importance = dict(zip(feature_names, sell_model.feature_importances_))
    print(f"✅ SELL model trained on {len(sell_trades)} trades (win rate: {sell_win_rate:.1f}%)")
    print(f"   Feature importance: {sell_importance}")

# Train BUY model if enough data
buy_model = None
buy_importance = None
if len(buy_trades) >= 30:
    X_buy, y_buy = extract_features(buy_trades)
    buy_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    buy_model.fit(X_buy, y_buy)
    joblib.dump(buy_model, "model_buy.pkl")
    buy_win_rate = sum(y_buy)/len(y_buy)*100
    buy_importance = dict(zip(feature_names, buy_model.feature_importances_))
    print(f"✅ BUY model trained on {len(buy_trades)} trades (win rate: {buy_win_rate:.1f}%)")
    print(f"   Feature importance: {buy_importance}")

# Save metadata (including feature importance)
overall_win_rate = sum(1 for t in trades if t["outcome"]=="win")/len(trades)*100
metadata = {
    "last_trained": datetime.utcnow().isoformat(),
    "total_trades": len(trades),
    "buy_trades": len(buy_trades),
    "sell_trades": len(sell_trades),
    "overall_win_rate": overall_win_rate,
    "sell_feature_importance": sell_importance,
    "buy_feature_importance": buy_importance,
}
if sell_model:
    metadata["sell_model_win_rate"] = sell_win_rate
if buy_model:
    metadata["buy_model_win_rate"] = buy_win_rate

with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("🎉 Training complete. Metadata saved.")
