"""
retrain_scheduler.py — Background auto-retrainer for stock prediction models.

Runs as a daemon thread inside the Flask app. Periodically checks all models
for staleness and retrains any that are older than MODEL_MAX_AGE_DAYS.

Key design decisions:
  - Runs in a background thread (not a separate process) so it shares
    the same Python environment and can update models in-place.
  - Retrains one stock at a time to avoid overloading the server.
  - Uses the same feature engineering and hyperparameters as train_models.py.
  - Updates model_metadata.json after each successful retrain.
  - Thread-safe: uses a lock so multiple retrain requests don't collide.
"""

import os
import time
import threading
import warnings
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score

from model_metadata import (
    set_model_metadata,
    get_stale_tickers,
    get_model_age_days,
    MODEL_MAX_AGE_DAYS,
)

warnings.filterwarnings("ignore")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
CHECK_INTERVAL_HOURS = 6  # How often to check for stale models
MIN_ROWS = 500            # Minimum rows before feature engineering
MIN_ROWS_AFTER = 200      # Minimum rows after feature engineering
PREDICTION_THRESHOLD = 0.55

# Predictors — must match model.py and train_models.py exactly
PREDICTORS = [
    "close_ratio_2",  "close_ratio_5",  "close_ratio_10",
    "close_ratio_21", "close_ratio_63", "close_ratio_252",
    "trend_2",  "trend_5",  "trend_10",
    "trend_21", "trend_63", "trend_252",
    "rsi_14", "macd_signal",
    "vol_ratio", "daily_return", "volatility_20", "hl_spread",
]

# Lock to prevent concurrent retraining of the same model
_retrain_lock = threading.Lock()

# Status tracking for the admin endpoint
_scheduler_status = {
    "running": False,
    "last_check": None,
    "currently_retraining": None,
    "retrained_count": 0,
    "failed_count": 0,
    "last_error": None,
}


def get_scheduler_status() -> dict:
    """Return current scheduler status for the admin endpoint."""
    return _scheduler_status.copy()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering — identical to model.py and train_models.py.
    Computes rolling ratios, RSI, MACD, volume ratio, volatility, etc.
    """
    df = df.copy()
    for horizon in [2, 5, 10, 21, 63, 252]:
        rolling_close = df["Close"].rolling(horizon).mean()
        df[f"close_ratio_{horizon}"] = df["Close"] / rolling_close
        df[f"trend_{horizon}"] = (
            df["Close"].shift(1).rolling(horizon).apply(
                lambda x: (x > x.shift(1)).sum(), raw=False
            ) / horizon
        )

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df["macd_signal"] = macd - macd.ewm(span=9, adjust=False).mean()

    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["daily_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["volatility_20"] = df["daily_return"].rolling(20).std()
    df["hl_spread"] = (df["High"] - df["Low"]) / df["Close"]
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary target: 1 if tomorrow's close > today's close."""
    df = df.copy()
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df


def _ticker_to_filename(ticker: str) -> str:
    """Convert ticker to safe filename, matching train_models.py convention."""
    safe = ticker.replace(".", "_").replace("&", "AND")
    return f"{safe}_model.pkl"


def retrain_single_model(ticker: str) -> dict:
    """
    Retrain the model for a single stock ticker.

    Downloads the latest data, engineers features, trains a new
    RandomForest model, evaluates it, and saves the .pkl file.

    Returns a result dict with status, metrics, and any errors.
    Thread-safe via _retrain_lock.
    """
    with _retrain_lock:
        _scheduler_status["currently_retraining"] = ticker
        model_path = os.path.join(MODELS_DIR, _ticker_to_filename(ticker))

        try:
            # Download latest data
            print(f"[RETRAIN] Downloading data for {ticker}...")
            data = yf.download(ticker, period="max", progress=False, timeout=30)

            if data is None or data.empty:
                _scheduler_status["failed_count"] += 1
                _scheduler_status["last_error"] = f"{ticker}: No data returned"
                return {"ticker": ticker, "status": "FAILED", "error": "No data returned"}

            # Fix MultiIndex columns (yfinance quirk)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data = data.loc[:, ~data.columns.duplicated()]

            required_cols = {"Open", "High", "Low", "Close", "Volume"}
            if not required_cols.issubset(set(data.columns)):
                missing = required_cols - set(data.columns)
                _scheduler_status["failed_count"] += 1
                return {"ticker": ticker, "status": "FAILED", "error": f"Missing columns: {missing}"}

            data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
            data.dropna(inplace=True)

            if len(data) < MIN_ROWS:
                _scheduler_status["failed_count"] += 1
                return {"ticker": ticker, "status": "FAILED", "error": f"Only {len(data)} rows"}

            # Feature engineering
            data = add_features(data)
            data = add_target(data)
            data.dropna(inplace=True)

            if len(data) < MIN_ROWS_AFTER:
                _scheduler_status["failed_count"] += 1
                return {"ticker": ticker, "status": "FAILED", "error": f"Only {len(data)} rows after features"}

            # Train/test split (80/20)
            split_idx = int(len(data) * 0.8)
            train = data.iloc[:split_idx]
            test = data.iloc[split_idx:]

            # Train model — same hyperparameters as train_models.py
            print(f"[RETRAIN] Training model for {ticker} ({len(train)} train rows)...")
            model = RandomForestClassifier(
                n_estimators=300,
                min_samples_split=100,
                max_depth=15,
                random_state=42,
                n_jobs=1,
                class_weight="balanced",
            )
            model.fit(train[PREDICTORS], train["Target"])

            # Evaluate
            preds = model.predict(test[PREDICTORS])
            proba = model.predict_proba(test[PREDICTORS])
            thresh = (proba[:, 1] >= PREDICTION_THRESHOLD).astype(int)

            acc = accuracy_score(test["Target"], preds)
            prec = precision_score(test["Target"], thresh, zero_division=0)

            # Save model
            os.makedirs(MODELS_DIR, exist_ok=True)
            joblib.dump(model, model_path)

            # Update metadata
            data_start = str(data.index[0].date()) if hasattr(data.index[0], 'date') else str(data.index[0])
            data_end = str(data.index[-1].date()) if hasattr(data.index[-1], 'date') else str(data.index[-1])

            set_model_metadata(
                ticker=ticker,
                accuracy=acc,
                precision=prec,
                data_start=data_start,
                data_end=data_end,
                rows=len(data),
            )

            _scheduler_status["retrained_count"] += 1
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"[RETRAIN] ✅ {ticker} done — acc={acc:.4f}, prec={prec:.4f}, {size_mb:.1f}MB")

            # Clear model from cache so next prediction loads the fresh model
            try:
                from model import _model_cache
                if ticker in _model_cache:
                    del _model_cache[ticker]
            except ImportError:
                pass

            return {
                "ticker": ticker,
                "status": "OK",
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "rows": len(data),
                "size_mb": round(size_mb, 1),
            }

        except Exception as e:
            _scheduler_status["failed_count"] += 1
            _scheduler_status["last_error"] = f"{ticker}: {str(e)}"
            print(f"[RETRAIN] ❌ {ticker} failed: {e}")
            return {"ticker": ticker, "status": "FAILED", "error": str(e)}

        finally:
            _scheduler_status["currently_retraining"] = None


def _scheduler_loop(all_tickers: list[str]):
    """
    Main loop for the background scheduler.
    Checks for stale models every CHECK_INTERVAL_HOURS and retrains them.
    """
    _scheduler_status["running"] = True
    print(f"[RETRAIN SCHEDULER] Started. Checking every {CHECK_INTERVAL_HOURS}h, "
          f"retraining models older than {MODEL_MAX_AGE_DAYS} days.")

    while True:
        try:
            _scheduler_status["last_check"] = datetime.now(timezone.utc).isoformat()

            # Find stale models
            stale = get_stale_tickers(all_tickers)

            if stale:
                print(f"[RETRAIN SCHEDULER] Found {len(stale)} stale model(s): "
                      f"{', '.join(stale[:5])}{'...' if len(stale) > 5 else ''}")

                for ticker in stale:
                    retrain_single_model(ticker)
                    # Small delay between retrains to be gentle on yfinance API
                    time.sleep(5)
            else:
                print(f"[RETRAIN SCHEDULER] All models are fresh (< {MODEL_MAX_AGE_DAYS} days old).")

        except Exception as e:
            print(f"[RETRAIN SCHEDULER] Error in check loop: {e}")
            _scheduler_status["last_error"] = str(e)

        # Sleep until next check
        time.sleep(CHECK_INTERVAL_HOURS * 3600)


def start_scheduler(all_tickers: list[str]):
    """
    Start the background retraining scheduler as a daemon thread.
    Called once from app.py on startup.
    """
    thread = threading.Thread(
        target=_scheduler_loop,
        args=(all_tickers,),
        name="ModelRetrainScheduler",
        daemon=True,  # Daemon thread — dies when main app exits
    )
    thread.start()
    print(f"[RETRAIN SCHEDULER] Background thread started (checking every {CHECK_INTERVAL_HOURS}h)")
