"""
train_models.py — Train individual RandomForest models for NIFTY 50 stocks.

Run this once to generate all models:
    python train_models.py

Models are saved to  ./models/<TICKER>_model.pkl
"""

import os
import sys
import io
import time
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score

warnings.filterwarnings("ignore")

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ─────────────────────────────────────────────────────────────────────
# NIFTY 50 constituents (Yahoo Finance tickers)
# ─────────────────────────────────────────────────────────────────────
NIFTY_50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
    "SUNPHARMA.NS", "BAJFINANCE.NS", "WIPRO.NS", "ULTRACEMCO.NS", "ONGC.NS",
    "NTPC.NS", "TATAMOTORS.NS", "JSWSTEEL.NS", "POWERGRID.NS", "M&M.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "TATASTEEL.NS", "HCLTECH.NS", "BAJAJFINSV.NS",
    "COALINDIA.NS", "TECHM.NS", "INDUSINDBK.NS", "HINDALCO.NS", "DRREDDY.NS",
    "CIPLA.NS", "BPCL.NS", "EICHERMOT.NS", "DIVISLAB.NS", "BRITANNIA.NS",
    "APOLLOHOSP.NS", "GRASIM.NS", "NESTLEIND.NS", "SBILIFE.NS", "HEROMOTOCO.NS",
    "TATACONSUM.NS", "BAJAJ-AUTO.NS", "HDFCLIFE.NS", "SHRIRAMFIN.NS", "LTIM.NS",
]

# Human-readable names for display
STOCK_NAMES = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank",
    "INFY.NS": "Infosys",
    "ICICIBANK.NS": "ICICI Bank",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ITC.NS": "ITC Limited",
    "SBIN.NS": "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "LT.NS": "Larsen & Toubro",
    "AXISBANK.NS": "Axis Bank",
    "ASIANPAINT.NS": "Asian Paints",
    "MARUTI.NS": "Maruti Suzuki",
    "TITAN.NS": "Titan Company",
    "SUNPHARMA.NS": "Sun Pharma",
    "BAJFINANCE.NS": "Bajaj Finance",
    "WIPRO.NS": "Wipro",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "ONGC.NS": "ONGC",
    "NTPC.NS": "NTPC Limited",
    "TATAMOTORS.NS": "Tata Motors",
    "JSWSTEEL.NS": "JSW Steel",
    "POWERGRID.NS": "Power Grid Corp",
    "M&M.NS": "Mahindra & Mahindra",
    "ADANIENT.NS": "Adani Enterprises",
    "ADANIPORTS.NS": "Adani Ports",
    "TATASTEEL.NS": "Tata Steel",
    "HCLTECH.NS": "HCL Technologies",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "COALINDIA.NS": "Coal India",
    "TECHM.NS": "Tech Mahindra",
    "INDUSINDBK.NS": "IndusInd Bank",
    "HINDALCO.NS": "Hindalco Industries",
    "DRREDDY.NS": "Dr. Reddy's Labs",
    "CIPLA.NS": "Cipla",
    "BPCL.NS": "Bharat Petroleum",
    "EICHERMOT.NS": "Eicher Motors",
    "DIVISLAB.NS": "Divi's Laboratories",
    "BRITANNIA.NS": "Britannia Industries",
    "APOLLOHOSP.NS": "Apollo Hospitals",
    "GRASIM.NS": "Grasim Industries",
    "NESTLEIND.NS": "Nestle India",
    "SBILIFE.NS": "SBI Life Insurance",
    "HEROMOTOCO.NS": "Hero MotoCorp",
    "TATACONSUM.NS": "Tata Consumer",
    "BAJAJ-AUTO.NS": "Bajaj Auto",
    "HDFCLIFE.NS": "HDFC Life Insurance",
    "SHRIRAMFIN.NS": "Shriram Finance",
    "LTIM.NS": "LTIMindtree",
}

# ─────────────────────────────────────────────────────────────────────
# Feature engineering (same pipeline as model.py)
# ─────────────────────────────────────────────────────────────────────
PREDICTORS = [
    "close_ratio_2", "close_ratio_5", "close_ratio_10",
    "close_ratio_21", "close_ratio_63", "close_ratio_252",
    "trend_2", "trend_5", "trend_10",
    "trend_21", "trend_63", "trend_252",
    "rsi_14", "macd_signal",
    "vol_ratio", "daily_return", "volatility_20", "hl_spread",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical + rolling features. No future leakage."""
    df = df.copy()

    # Rolling close ratio & trend
    for horizon in [2, 5, 10, 21, 63, 252]:
        rolling_close = df["Close"].rolling(horizon).mean()
        df[f"close_ratio_{horizon}"] = df["Close"] / rolling_close
        df[f"trend_{horizon}"] = (
            df["Close"].shift(1).rolling(horizon).apply(
                lambda x: (x > x.shift(1)).sum(), raw=False
            ) / horizon
        )

    # RSI (14-day)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD signal
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal_line = macd.ewm(span=9, adjust=False).mean()
    df["macd_signal"] = macd - signal_line

    # Volume ratio
    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # Daily log return & rolling volatility
    df["daily_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["volatility_20"] = df["daily_return"].rolling(20).std()

    # High-low spread
    df["hl_spread"] = (df["High"] - df["Low"]) / df["Close"]

    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Target = 1 if tomorrow's close > today's close."""
    df = df.copy()
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df


def train_single_stock(ticker: str, models_dir: str) -> dict:
    """
    Download data, engineer features, train model, save, and return metrics.
    """
    safe_name = ticker.replace(".", "_").replace("&", "AND")
    model_path = os.path.join(models_dir, f"{safe_name}_model.pkl")

    print(f"\n{'='*60}")
    print(f"  Training: {ticker}  ({STOCK_NAMES.get(ticker, ticker)})")
    print(f"{'='*60}")

    # Download
    try:
        data = yf.download(ticker, period="max", progress=False)
    except Exception as e:
        print(f"  [X] Download failed: {e}")
        return {"ticker": ticker, "status": "FAILED", "error": str(e)}

    if data.empty:
        print(f"  [X] No data returned")
        return {"ticker": ticker, "status": "FAILED", "error": "No data"}

    # Fix MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
    data.dropna(inplace=True)

    if len(data) < 500:
        print(f"  [X] Only {len(data)} rows (need at least 500)")
        return {"ticker": ticker, "status": "FAILED", "error": "Insufficient data"}

    print(f"  [OK] Downloaded {len(data)} rows  ({data.index[0].date()} -> {data.index[-1].date()})")

    # Feature engineering
    data = add_features(data)
    data = add_target(data)
    data.dropna(inplace=True)

    if len(data) < 200:
        print(f"  [X] Only {len(data)} rows after feature engineering")
        return {"ticker": ticker, "status": "FAILED", "error": "Insufficient data after features"}

    # Time-series split (no shuffle — last 20% is test)
    split_idx = int(len(data) * 0.8)
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]

    print(f"  [OK] Train: {len(train)} rows  |  Test: {len(test)} rows")

    # Train
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_split=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(train[PREDICTORS], train["Target"])

    # Evaluate
    preds = model.predict(test[PREDICTORS])
    proba = model.predict_proba(test[PREDICTORS])

    # Use threshold for better precision
    threshold_preds = (proba[:, 1] >= 0.55).astype(int)

    acc = accuracy_score(test["Target"], preds)
    prec_default = precision_score(test["Target"], preds, zero_division=0)
    prec_threshold = precision_score(test["Target"], threshold_preds, zero_division=0)

    print(f"  [OK] Accuracy:  {acc:.2%}")
    print(f"  [OK] Precision (default):    {prec_default:.2%}")
    print(f"  [OK] Precision (threshold):  {prec_threshold:.2%}")

    # Save
    joblib.dump(model, model_path)
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  [OK] Saved: {model_path}  ({size_mb:.1f} MB)")

    return {
        "ticker": ticker,
        "status": "OK",
        "rows": len(data),
        "accuracy": f"{acc:.2%}",
        "precision": f"{prec_threshold:.2%}",
        "size_mb": f"{size_mb:.1f}",
    }


def main():
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print("  NIFTY 50 Stock Model Training Pipeline")
    print("  Training individual models for 50 stocks")
    print("=" * 60)
    print(f"\nModels directory: {models_dir}")
    print(f"Stocks to train:  {len(NIFTY_50_TICKERS)}\n")

    start = time.time()
    results = []

    for i, ticker in enumerate(NIFTY_50_TICKERS, 1):
        print(f"\n[{i}/{len(NIFTY_50_TICKERS)}]", end="")
        result = train_single_stock(ticker, models_dir)
        results.append(result)

    elapsed = time.time() - start

    # Summary
    print("\n\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)

    ok = [r for r in results if r["status"] == "OK"]
    failed = [r for r in results if r["status"] == "FAILED"]

    print(f"\n  Successful:  {len(ok)} / {len(NIFTY_50_TICKERS)}")
    print(f"  Failed:      {len(failed)} / {len(NIFTY_50_TICKERS)}")
    print(f"  Total time:  {elapsed:.0f}s  ({elapsed/60:.1f} min)\n")

    if ok:
        print(f"  {'Ticker':<20} {'Rows':>8} {'Accuracy':>10} {'Precision':>10} {'Size':>8}")
        print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
        for r in ok:
            print(f"  {r['ticker']:<20} {r['rows']:>8} {r['accuracy']:>10} {r['precision']:>10} {r['size_mb']:>6} MB")

    if failed:
        print(f"\n  Failed stocks:")
        for r in failed:
            print(f"    [X] {r['ticker']}: {r['error']}")

    print(f"\n{'='*60}")
    print("  Done! Models saved to:", models_dir)
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
