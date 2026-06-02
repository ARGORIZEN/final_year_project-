"""
model_metadata.py — Tracks training timestamps and metrics for each model.

Stores a JSON file alongside the .pkl models so the app knows:
  - When each model was last trained
  - What accuracy/precision it achieved
  - What date range the training data covered

This lets us detect stale models and trigger automatic retraining.
"""

import json
import os
from datetime import datetime, timezone

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
METADATA_FILE = os.path.join(MODELS_DIR, "model_metadata.json")

# Models older than this many days are considered stale
MODEL_MAX_AGE_DAYS = 7


def _load_metadata() -> dict:
    """Load the metadata JSON file. Returns empty dict if not found."""
    if not os.path.exists(METADATA_FILE):
        return {}
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_metadata(data: dict):
    """Save metadata dict to JSON file."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_model_metadata(ticker: str) -> dict | None:
    """
    Get metadata for a specific ticker.
    Returns dict with keys: last_trained, accuracy, precision, data_start, data_end
    Returns None if no metadata exists for this ticker.
    """
    data = _load_metadata()
    return data.get(ticker)


def set_model_metadata(ticker: str, accuracy: float = 0.0,
                       precision: float = 0.0, data_start: str = "",
                       data_end: str = "", rows: int = 0):
    """
    Save/update metadata for a ticker after training.
    Automatically sets last_trained to current UTC time.
    """
    data = _load_metadata()
    data[ticker] = {
        "last_trained": datetime.now(timezone.utc).isoformat(),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "data_start": data_start,
        "data_end": data_end,
        "rows": rows,
    }
    _save_metadata(data)


def get_model_age_days(ticker: str) -> float | None:
    """
    Returns the age of the model in days.
    Returns None if no metadata exists (model was trained before metadata tracking).
    """
    meta = get_model_metadata(ticker)
    if meta is None or "last_trained" not in meta:
        return None

    try:
        trained_at = datetime.fromisoformat(meta["last_trained"])
        now = datetime.now(timezone.utc)
        age = (now - trained_at).total_seconds() / 86400  # seconds in a day
        return round(age, 1)
    except (ValueError, TypeError):
        return None


def is_model_stale(ticker: str) -> bool:
    """
    Returns True if the model is older than MODEL_MAX_AGE_DAYS.
    Returns False if no metadata exists (we treat unknown-age models as fresh
    until the system starts tracking them).
    """
    age = get_model_age_days(ticker)
    if age is None:
        return False  # No metadata yet — don't flag as stale
    return age > MODEL_MAX_AGE_DAYS


def get_stale_tickers(all_tickers: list[str]) -> list[str]:
    """
    Returns a list of tickers whose models are older than MODEL_MAX_AGE_DAYS.
    Also includes tickers that have a .pkl file but no metadata entry
    (these need to be registered).
    """
    stale = []
    for ticker in all_tickers:
        safe = ticker.replace(".", "_").replace("&", "AND")
        pkl_path = os.path.join(MODELS_DIR, f"{safe}_model.pkl")
        if not os.path.exists(pkl_path):
            continue  # No model file — skip

        age = get_model_age_days(ticker)
        if age is None:
            # Model exists but no metadata — register it with current time
            # so it starts aging from now
            _register_existing_model(ticker)
            continue
        if age > MODEL_MAX_AGE_DAYS:
            stale.append(ticker)
    return stale


def _register_existing_model(ticker: str):
    """
    For models that existed before the metadata system,
    register them with the current timestamp so they start
    aging from now (won't immediately be marked stale).
    """
    data = _load_metadata()
    if ticker not in data:
        data[ticker] = {
            "last_trained": datetime.now(timezone.utc).isoformat(),
            "accuracy": 0.0,
            "precision": 0.0,
            "data_start": "",
            "data_end": "",
            "rows": 0,
            "note": "Pre-existing model, registered when metadata system started",
        }
        _save_metadata(data)


def get_all_metadata_summary() -> dict:
    """
    Returns a summary: total models, stale count, freshest/oldest model dates.
    Useful for the admin status endpoint.
    """
    data = _load_metadata()
    if not data:
        return {"total": 0, "stale": 0, "oldest": None, "newest": None}

    ages = []
    for ticker, meta in data.items():
        try:
            trained_at = datetime.fromisoformat(meta["last_trained"])
            age = (datetime.now(timezone.utc) - trained_at).total_seconds() / 86400
            ages.append((ticker, age))
        except (ValueError, TypeError, KeyError):
            pass

    stale_count = sum(1 for _, age in ages if age > MODEL_MAX_AGE_DAYS)
    oldest = max(ages, key=lambda x: x[1]) if ages else None
    newest = min(ages, key=lambda x: x[1]) if ages else None

    return {
        "total": len(data),
        "stale": stale_count,
        "oldest_ticker": oldest[0] if oldest else None,
        "oldest_age_days": round(oldest[1], 1) if oldest else None,
        "newest_ticker": newest[0] if newest else None,
        "newest_age_days": round(newest[1], 1) if newest else None,
    }
