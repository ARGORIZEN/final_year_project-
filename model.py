"""
model.py — Dynamic multi-stock prediction engine for NIFTY 50.
Models are loaded on demand and cached in memory.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
PREDICTION_THRESHOLD = 0.55

NIFTY_50_STOCKS = {
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
}

PREDICTORS = [
    "close_ratio_2",  "close_ratio_5",  "close_ratio_10",
    "close_ratio_21", "close_ratio_63", "close_ratio_252",
    "trend_2",  "trend_5",  "trend_10",
    "trend_21", "trend_63", "trend_252",
    "rsi_14", "macd_signal",
    "vol_ratio", "daily_return", "volatility_20", "hl_spread",
]

_model_cache = {}


def _ticker_to_filename(ticker):
    safe = ticker.replace(".", "_").replace("&", "AND")
    return f"{safe}_model.pkl"


def _load_model(ticker):
    if ticker in _model_cache:
        return _model_cache[ticker]
    filename = _ticker_to_filename(ticker)
    model_path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(model_path):
        return None
    model = joblib.load(model_path)
    _model_cache[ticker] = model
    return model


def get_supported_stocks():
    stocks = []
    for ticker, name in NIFTY_50_STOCKS.items():
        short = ticker.replace(".NS", "")
        stocks.append({"ticker": ticker, "short": short, "name": name,
                        "label": f"{short} — {name}"})
    return stocks


def add_features(df):
    df = df.copy()
    for horizon in [2, 5, 10, 21, 63, 252]:
        rolling_close = df["Close"].rolling(horizon).mean()
        df[f"close_ratio_{horizon}"] = df["Close"] / rolling_close
        df[f"trend_{horizon}"] = (
            df["Close"].shift(1).rolling(horizon).apply(
                lambda x: (x > x.shift(1)).sum(), raw=False) / horizon)

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal_line = macd.ewm(span=9, adjust=False).mean()
    df["macd_signal"] = macd - signal_line

    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["daily_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["volatility_20"] = df["daily_return"].rolling(20).std()
    df["hl_spread"] = (df["High"] - df["Low"]) / df["Close"]
    return df


def predict_stock(stock):
    try:
        stock = stock.strip().upper()
        if not stock.startswith("^") and "." not in stock:
            stock = stock + ".NS"

        model = _load_model(stock)
        if model is None:
            return {"error": "unsupported", "ticker": stock}

        data = yf.download(stock, period="max", progress=False)
        if data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
        data.dropna(inplace=True)

        if len(data) < 300:
            return None

        data = add_features(data)
        data.dropna(inplace=True)

        if len(data) < 2:
            return None

        missing = [p for p in PREDICTORS if p not in data.columns]
        if missing:
            return None

        latest_data = data[PREDICTORS].tail(1)
        probabilities = model.predict_proba(latest_data)[0]
        up_prob = probabilities[1]
        prediction = 1 if up_prob >= PREDICTION_THRESHOLD else 0
        confidence = up_prob if prediction == 1 else probabilities[0]
        trend = "Bullish" if prediction == 1 else "Bearish"

        latest = data.iloc[-1]
        rsi_val = float(latest["rsi_14"])
        macd_val = float(latest["macd_signal"])
        vol_val = float(latest["volatility_20"])

        if rsi_val > 70:
            rsi_zone = "Overbought"
        elif rsi_val < 30:
            rsi_zone = "Oversold"
        else:
            rsi_zone = "Neutral"

        macd_trend = "Bullish" if macd_val > 0 else "Bearish"

        if vol_val > 0.03:
            vol_level = "High"
        elif vol_val > 0.015:
            vol_level = "Medium"
        else:
            vol_level = "Low"

        history = data.tail(60)
        chart_data = {
            "dates": history.index.strftime('%Y-%m-%d').tolist(),
            "prices": history['Close'].tolist()
        }

        # Calculate 30-day and quarterly changes
        price_30d_ago = float(data['Close'].iloc[-30]) if len(data) >= 30 else float(data['Close'].iloc[0])
        price_90d_ago = float(data['Close'].iloc[-90]) if len(data) >= 90 else float(data['Close'].iloc[0])
        current_price = float(data['Close'].iloc[-1])
        change_30d = round(((current_price - price_30d_ago) / price_30d_ago) * 100, 1)
        change_90d = round(((current_price - price_90d_ago) / price_90d_ago) * 100, 1)
        direction = "Up" if change_30d >= 0 else "Down"

        # Build "why up/down" analysis bullet points
        short_name = NIFTY_50_STOCKS.get(stock, stock.replace('.NS', ''))
        ticker_short = stock.replace('.NS', '')
        analysis_bullets = []

        # Price movement summary
        analysis_bullets.append(
            f"{ticker_short} stock {'gained' if change_30d >= 0 else 'declined'} "
            f"{'+' if change_30d >= 0 else ''}{change_30d}% over the past 30 days, "
            f"{'rising' if change_30d >= 0 else 'falling'} from ₹{price_30d_ago:,.0f} to ₹{current_price:,.0f}."
        )

        # RSI insight
        if rsi_val > 70:
            analysis_bullets.append(
                f"RSI at {rsi_val:.1f} signals overbought conditions — the stock may be due for a pullback "
                "as buying momentum becomes extended."
            )
        elif rsi_val < 30:
            analysis_bullets.append(
                f"RSI at {rsi_val:.1f} signals oversold conditions — the stock may be approaching a reversal "
                "as selling pressure appears exhausted."
            )
        else:
            analysis_bullets.append(
                f"RSI at {rsi_val:.1f} remains in neutral territory, indicating balanced buying and selling pressure."
            )

        # MACD insight
        if macd_val > 0:
            analysis_bullets.append(
                f"MACD histogram is positive ({macd_val:+.2f}), confirming bullish momentum with the signal line "
                "trending above the baseline."
            )
        else:
            analysis_bullets.append(
                f"MACD histogram is negative ({macd_val:.2f}), indicating bearish momentum with the signal line "
                "trending below the baseline."
            )

        # Volatility insight
        if vol_level == "High":
            analysis_bullets.append(
                f"Volatility is elevated at {vol_val*100:.2f}%, suggesting large price swings and increased "
                "market uncertainty around the stock."
            )
        elif vol_level == "Low":
            analysis_bullets.append(
                f"Volatility is low at {vol_val*100:.2f}%, indicating stable price action and reduced risk."
            )

        # Volume insight
        vol_r = float(latest["vol_ratio"])
        if vol_r > 1.5:
            analysis_bullets.append(
                f"Trading volume is {vol_r:.1f}x above the 20-day average, indicating strong institutional "
                "interest and conviction in the current move."
            )
        elif vol_r < 0.5:
            analysis_bullets.append(
                f"Volume is {vol_r:.1f}x below average — the current trend lacks strong participation "
                "and may be prone to reversal."
            )

        # Quarter summary
        if abs(change_90d) > 1:
            analysis_bullets.append(
                f"Over the past quarter, the stock {'advanced' if change_90d >= 0 else 'declined'} "
                f"{'+' if change_90d >= 0 else ''}{change_90d}% from ₹{price_90d_ago:,.0f}."
            )

        return {
            "trend": trend,
            "confidence": f"{confidence:.2%}",
            "current_price": current_price,
            "chart_data": chart_data,
            "stock_name": NIFTY_50_STOCKS.get(stock, stock),
            "ticker": stock,
            "rsi": round(rsi_val, 1),
            "rsi_zone": rsi_zone,
            "macd": round(macd_val, 2),
            "macd_trend": macd_trend,
            "volatility": round(vol_val * 100, 2),
            "vol_level": vol_level,
            "daily_return_pct": round(float(latest["daily_return"]) * 100, 2),
            "day_high": round(float(latest["High"]), 2),
            "day_low": round(float(latest["Low"]), 2),
            "volume": int(latest["Volume"]),
            "vol_ratio": round(vol_r, 2),
            "change_30d": change_30d,
            "change_90d": change_90d,
            "direction": direction,
            "analysis_bullets": analysis_bullets,
        }
    except Exception as e:
        print(f"DEBUG: Error in predict_stock: {e}")
        import traceback
        traceback.print_exc()
        return None