"""
train_models_500.py — Train individual RandomForest models for Top 500 Indian stocks.

Upgrades over the NIFTY-50 version:
  - 500 stocks across NIFTY 50, NIFTY Next 50, NIFTY Midcap 150, NIFTY Smallcap 250
  - Parallel training using ProcessPoolExecutor (configurable workers)
  - Resume support: skips tickers whose .pkl already exists (use --retrain to force)
  - Per-stock retry logic (up to 3 attempts) with back-off on download errors
  - CSV progress log saved after every stock so you never lose results
  - Memory-efficient: each worker process is independent and releases RAM after saving

Run:
    python train_models_500.py                  # train all, skip already-done
    python train_models_500.py --retrain        # force retrain every model
    python train_models_500.py --workers 4      # set parallel workers (default: 4)
    python train_models_500.py --tickers-only   # just print the ticker list and exit
"""

import os
import sys
import io
import time
import warnings
import argparse
import csv
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score

from model_metadata import set_model_metadata

warnings.filterwarnings("ignore")

# Fix Windows console encoding (no-op on Linux/Mac)
if hasattr(sys.stdout, "buffer"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────
# TOP 500 INDIAN STOCK TICKERS (Yahoo Finance .NS suffix)
# Covers: NIFTY 50 + NIFTY Next 50 + NIFTY Midcap 150 + NIFTY Smallcap 250
# ─────────────────────────────────────────────────────────────────────
ALL_500_TICKERS = [
    # ── NIFTY 50 ──────────────────────────────────────────────────────
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

    # ── NIFTY Next 50 ─────────────────────────────────────────────────
    "ADANIGREEN.NS", "ADANITRANS.NS", "AMBUJACEM.NS", "AUROPHARMA.NS", "BAJAJHLDNG.NS",
    "BANKBARODA.NS", "BEL.NS", "BERGEPAINT.NS", "BIOCON.NS", "BOSCHLTD.NS",
    "CANBK.NS", "CHOLAFIN.NS", "COLPAL.NS", "DABUR.NS", "DLF.NS",
    "GAIL.NS", "GODREJCP.NS", "HAVELLS.NS", "ICICIPRULI.NS", "ICICIGI.NS",
    "INDUSTOWER.NS", "IOC.NS", "IRCTC.NS", "JINDALSTEL.NS", "LUPIN.NS",
    "MCDOWELL-N.NS", "MUTHOOTFIN.NS", "NAUKRI.NS", "NMDC.NS", "PAGEIND.NS",
    "PERSISTENT.NS", "PETRONET.NS", "PIIND.NS", "PIDILITIND.NS", "PNB.NS",
    "POLYCAB.NS", "RECLTD.NS", "SAIL.NS", "SIEMENS.NS", "SRF.NS",
    "TORNTPHARM.NS", "TRENT.NS", "TVSMOTOR.NS", "UBL.NS", "UNITDSPR.NS",
    "UPL.NS", "VEDL.NS", "VOLTAS.NS", "ZYDUSLIFE.NS", "HAL.NS",

    # ── NIFTY Midcap 150 ──────────────────────────────────────────────
    "AARTIIND.NS", "ABB.NS", "ABCAPITAL.NS", "ABFRL.NS", "ACC.NS",
    "APLAPOLLO.NS", "ASTRAL.NS", "ATUL.NS", "AUBANK.NS", "BALKRISIND.NS",
    "BATAINDIA.NS", "BHEL.NS", "BSOFT.NS", "CAMS.NS", "CANFINHOME.NS",
    "CDSL.NS", "CESC.NS", "CGPOWER.NS", "CONCOR.NS", "COROMANDEL.NS",
    "CROMPTON.NS", "CUMMINSIND.NS", "DEEPAKNTR.NS", "DIXON.NS", "ELGIEQUIP.NS",
    "EMAMILTD.NS", "ESCORTS.NS", "EXIDEIND.NS", "FEDERALBNK.NS", "FORTIS.NS",
    "GLAXO.NS", "GMRAIRPORT.NS", "GODREJIND.NS", "GODREJPROP.NS", "GRANULES.NS",
    "GSPL.NS", "GUJGASLTD.NS", "HFCL.NS", "HONAUT.NS", "IDFCFIRSTB.NS",
    "IGL.NS", "INDHOTEL.NS", "INDIAMART.NS", "INOXWIND.NS", "IRFC.NS",
    "ISEC.NS", "JKCEMENT.NS", "JUBLFOOD.NS", "KAJARIACER.NS", "KALYANKJIL.NS",
    "KANSAINER.NS", "KEC.NS", "KPITTECH.NS", "LALPATHLAB.NS", "LAURUSLABS.NS",
    "LICHSGFIN.NS", "LTTS.NS", "MARICO.NS", "MAXHEALTH.NS", "MCX.NS",
    "METROPOLIS.NS", "MFSL.NS", "MOTHERSON.NS", "MPHASIS.NS", "MRF.NS",
    "NATCOPHARM.NS", "NBCC.NS", "NCC.NS", "NHPC.NS", "NLCINDIA.NS",
    "NYKAA.NS", "OBEROIRLTY.NS", "OFSS.NS", "OIL.NS", "OLECTRA.NS",
    "PHOENIXLTD.NS", "PRESTIGE.NS", "PVRINOX.NS", "RAIN.NS", "RAJESHEXPO.NS",
    "RAMCOCEM.NS", "RITES.NS", "SBICARD.NS", "SCHAEFFLER.NS", "SHREECEM.NS",
    "SJVN.NS", "SKFINDIA.NS", "SOBHA.NS", "SONACOMS.NS", "STARHEALTH.NS",
    "SUMICHEM.NS", "SUNDARMFIN.NS", "SUNDRMFAST.NS", "SUPREMEIND.NS", "SYNGENE.NS",
    "TATACOMM.NS", "TATAELXSI.NS", "TATAINVEST.NS", "TATAPOWER.NS", "TEAMLEASE.NS",
    "THERMAX.NS", "TIINDIA.NS", "TIMKEN.NS", "TITAGARH.NS", "TORNTPOWER.NS",
    "TRIDENT.NS", "TRITURBINE.NS", "UBLLTD.NS", "UJJIVANSFB.NS", "VGUARD.NS",
    "VBL.NS", "WHIRLPOOL.NS", "WIPRO.NS", "WOCKPHARMA.NS", "ZEEL.NS",
    "ZENTEC.NS", "ZENSARTECH.NS", "3MINDIA.NS", "AARTIDRUGS.NS", "ABBOTINDIA.NS",
    "AEGISCHEM.NS", "AFFLE.NS", "AJANTPHARM.NS", "ALKEM.NS", "AMBER.NS",
    "ANGELONE.NS", "APOLLOTYRE.NS", "APTUS.NS", "ATUL.NS", "AVANTIFEED.NS",
    "BASF.NS", "BAYERCROP.NS", "BBTC.NS", "BLUEDART.NS", "BLUESTARCO.NS",
    "BRIGADE.NS", "BSE.NS", "CAMPUS.NS", "CARBORUNIV.NS", "CCL.NS",
    "CENTURYTEX.NS", "CHAMBLFERT.NS", "CHEMCON.NS", "CIGNITI.NS", "CLEAN.NS",

    # ── NIFTY Smallcap 250 (representative selection) ─────────────────
    "AAVAS.NS", "ACCELYA.NS", "ACE.NS", "ADANIPOWER.NS", "AEGISCHEM.NS",
    "AETHER.NS", "AGIIL.NS", "AGROPHOS.NS", "AIIL.NS", "AKZOINDIA.NS",
    "ALANKIT.NS", "ALEMBICLTD.NS", "ALICON.NS", "ALKYLAMINE.NS", "ALLCARGO.NS",
    "AMARAJABAT.NS", "AMJLAND.NS", "ANANTRAJ.NS", "ANDHRSUGAR.NS", "ANGELBRKG.NS",
    "ANSALAPI.NS", "APARINDS.NS", "APOLLOPIPE.NS", "ARCOTECH.NS", "ARFIN.NS",
    "ARIHANTCAP.NS", "ARMANFIN.NS", "ARROWGREEN.NS", "ARVIND.NS", "ARVINDFASN.NS",
    "ASAHIINDIA.NS", "ASALCBR.NS", "ASHIANA.NS", "ASHIMASYN.NS", "ASKAUTOLTD.NS",
    "ASMTEC.NS", "ATGL.NS", "ATIL.NS", "ATUL.NS", "AUROBINDO.NS",
    "AVTNPL.NS", "AXISGOLD.NS", "AZAD.NS", "BAJAJCON.NS", "BALAJITELE.NS",
    "BALMLAWRIE.NS", "BANARISUG.NS", "BANSWRAS.NS", "BARBEQUE.NS", "BASML.NS",
    "BDHL.NS", "BEML.NS", "BFUTILITIE.NS", "BHAGCHEM.NS", "BHANDARI.NS",
    "BHARATFORG.NS", "BHARATRAS.NS", "BIKAJI.NS", "BINDALAGRO.NS", "BIRLACORPN.NS",
    "BKMINDST.NS", "BOROLTD.NS", "BOROSIL.NS", "BPCL.NS", "BQLIND.NS",
    "CRAFTSMAN.NS", "CREATIVSER.NS", "CRISIL.NS", "CYIENTDLM.NS", "DALMIASUG.NS",
    "DATAMATICS.NS", "DBCORP.NS", "DCMSHRIRAM.NS", "DCXSYS.NS", "DOMS.NS",
    "DREDGECORP.NS", "DUROPLY.NS", "DYNAMATECH.NS", "EASEMYTRIP.NS", "EFCLTD.NS",
    "EIDPARRY.NS", "ELECON.NS", "ELGIEQUIP.NS", "EMKAYTOOLS.NS", "ENDURANCE.NS",
    "ENIL.NS", "EPIGRAL.NS", "EQUITASBNK.NS", "ESSARSHPNG.NS", "ETHOSLTD.NS",
    "EUROBATIND.NS", "EXLSERVICE.NS", "FAIRCHEM.NS", "FCSSOFT.NS", "FINEORG.NS",
    "FINOLEX.NS", "FINOPB.NS", "FLFL.NS", "FOODWORKS.NS", "FORCE.NS",
    "GANDHAR.NS", "GARFIBRES.NS", "GARNET.NS", "GEOJITFSL.NS", "GHCL.NS",
    "GLOBUSSPR.NS", "GLODYNE.NS", "GNFC.NS", "GPIL.NS", "GPTINFRA.NS",
    "GRAPHITE.NS", "GRAVITA.NS", "GREENPOWER.NS", "GREENPANEL.NS", "GRINFRA.NS",
    "GRSE.NS", "GTLINFRA.NS", "GUFICBIO.NS", "GULFOILLUB.NS", "HARDWYN.NS",
    "HATHWAY.NS", "HEMIPROP.NS", "HERITGFOOD.NS", "HFCL.NS", "HIKAL.NS",
    "HIL.NS", "HIRECT.NS", "HLEGLAS.NS", "HMT.NS", "HOMEFIRST.NS",
    "ICICIB22.NS", "IDEAFORGE.NS", "IGPL.NS", "IITL.NS", "IMAGICAA.NS",
    "IMFA.NS", "INDIACEM.NS", "INDIGOPNTS.NS", "INDORAMA.NS", "INDOSTAR.NS",
    "INNOVATORS.NS", "INSECTICID.NS", "INTELLECT.NS", "INVENTURE.NS", "IPCALAB.NS",
    "IPL.NS", "IRCON.NS", "IRB.NS", "ISFT.NS", "ITDC.NS",
    "J&KBANK.NS", "JAICORPLTD.NS", "JAMNAAUTO.NS", "JAYAGROGN.NS", "JAYNECOIND.NS",
    "JBCHEPHARM.NS", "JBMA.NS", "JKIL.NS", "JKLAKSHMI.NS", "JKPAPER.NS",
    "JKTYRE.NS", "JMFINANCIL.NS", "JPPOWER.NS", "JSWENERGY.NS", "JTEKTINDIA.NS",
    "JUBLINGREA.NS", "JUNIPR.NS", "JUSTDIAL.NS", "JYOTHYLAB.NS", "JYOTISTRUC.NS",
    "KABRAEXTRU.NS", "KALPATPOWR.NS", "KANORICHEM.NS", "KARNATAK.NS", "KAYNES.NS",
    "KESORAMIND.NS", "KFINTECH.NS", "KHAITANLTD.NS", "KHADIM.NS", "KILPEST.NS",
    "KINETIC.NS", "KITEX.NS", "KNRCON.NS", "KPIL.NS", "KRBL.NS",
    "KSCL.NS", "LATENTVIEW.NS", "LAXMIMACH.NS", "LEMONTREE.NS", "LGBBROSLTD.NS",
    "LLOYDSENGG.NS", "LNTFH.NS", "LOTUSEYE.NS", "LUXIND.NS", "MAHAPEXLTD.NS",
    "MAHINDCIE.NS", "MANAPPURAM.NS", "MANGLMCEM.NS", "MANINDS.NS", "MANKIND.NS",
    "MARATHON.NS", "MARKSANS.NS", "MASTEK.NS", "MATRIMONY.NS", "MAYURUNIQ.NS",
    "MEDPLUS.NS", "MIDHANI.NS", "MINDA.NS", "MITCON.NS", "MMTC.NS",
    "MOIL.NS", "MRPL.NS", "MSTCLTD.NS", "MUTHOOTCAP.NS", "NACLIND.NS",
    "NAGAFERT.NS", "NAVINFLUOR.NS", "NESCO.NS", "NETWORK18.NS", "NIACL.NS",
    "NIITLTD.NS", "NILE.NS", "NIRAJ.NS", "NOCIL.NS", "NUVOCO.NS",
    "OPTIEMUS.NS", "ORISSAMINE.NS", "OSIAJEE.NS", "PAGEIND.NS", "PALREDTEC.NS",
    "PANAMAPET.NS", "PATELENG.NS", "PATINTLOG.NS", "PCJEWELLER.NS", "PDSL.NS",
    "PENIND.NS", "PGINDUSTRY.NS", "PGIL.NS", "PHOENIXLTD.NS", "PILANIINVS.NS",
    "PLASTIBLEN.NS", "POKARNA.NS", "POLYMED.NS", "POONAWALLA.NS", "POWERMECH.NS",
    "PRAKASH.NS", "PRICOLLTD.NS", "PRINCEPIPE.NS", "PRISMJOINTS.NS", "PRIVISCL.NS",
    "PSPPROJECT.NS", "PTCIL.NS", "PURVA.NS", "QUESS.NS", "RADHIKJWE.NS",
    "RADICO.NS", "RAILTEL.NS", "RAJRATAN.NS", "RAJSREESUG.NS", "RALLIS.NS",
    "RAMCOIND.NS", "RATNAMANI.NS", "RAYMOND.NS", "RBA.NS", "RBLBANK.NS",
    "REDINGTON.NS", "REPCO.NS", "RESPONIND.NS", "RKFORGE.NS", "ROHLTD.NS",
    "RUPA.NS", "RUSHIL.NS", "SAFARI.NS", "SALZERELEC.NS", "SANOFI.NS",
    "SAPPHIRE.NS", "SARDA.NS", "SAREGAMA.NS", "SATIA.NS", "SBFC.NS",
    "SEPOWER.NS", "SGXNIFTY.NS", "SHAKTIPUMP.NS", "SHANTIGEAR.NS", "SHARDACROP.NS",
    "SHAREINDIA.NS", "SHILPAMED.NS", "SHREDIGCEM.NS", "SHYAMMETL.NS", "SIGIND.NS",
    "SKPMAR.NS", "SMLISUZU.NS", "SNOWMAN.NS", "SOLARA.NS", "SONATSOFTW.NS",
    "SOUTHBANK.NS", "SPANDANA.NS", "SPECIALITY.NS", "SPTL.NS", "SRINDLTH.NS",
    "SSWL.NS", "STAR.NS", "STCINDIA.NS", "STEELXIND.NS", "STYRENIX.NS",
    "SUBROS.NS", "SUDARSCHEM.NS", "SUGIND.NS", "SUKHJITS.NS", "SURYAROSNI.NS",
    "SUULD.NS", "SWSOLAR.NS", "SYMPHONY.NS", "SYNCOMF.NS", "TARSONS.NS",
    "TATAMETALI.NS", "TCIEXP.NS", "TCNSBRANDS.NS", "TDPOWERSYS.NS", "TEXRAIL.NS",
    "THYROCARE.NS", "TINPLATE.NS", "TIPSINDLTD.NS", "TIRUMALCHM.NS", "TMRVL.NS",
    "TNPETRO.NS", "TPLPLASTEH.NS", "TREEHOUSE.NS", "TRIL.NS", "TRIVENI.NS",
]

# Deduplicate while preserving order
seen = set()
ALL_500_TICKERS = [t for t in ALL_500_TICKERS if not (t in seen or seen.add(t))]

# ─────────────────────────────────────────────────────────────────────
# Feature engineering — identical pipeline for every model
# ─────────────────────────────────────────────────────────────────────
PREDICTORS = [
    "close_ratio_2", "close_ratio_5", "close_ratio_10",
    "close_ratio_21", "close_ratio_63", "close_ratio_252",
    "trend_2", "trend_5", "trend_10",
    "trend_21", "trend_63", "trend_252",
    "rsi_14", "macd_signal",
    "vol_ratio", "daily_return", "volatility_20", "hl_spread",
]

MAX_RETRIES = 3           # download retries per ticker
RETRY_DELAY = 5           # seconds between retries
MIN_ROWS = 500            # minimum rows before features
MIN_ROWS_AFTER = 200      # minimum rows after feature engineering
DEFAULT_WORKERS = 4       # parallel processes


def add_features(df: pd.DataFrame) -> pd.DataFrame:
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
    df = df.copy()
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────
# Single-stock training (runs in worker process)
# ─────────────────────────────────────────────────────────────────────
def train_single_stock(args: tuple) -> dict:
    """
    Designed to run in a subprocess. Returns a result dict.
    args = (ticker, models_dir, retrain)
    """
    ticker, models_dir, retrain = args
    safe_name = ticker.replace(".", "_").replace("&", "AND")
    model_path = os.path.join(models_dir, f"{safe_name}_model.pkl")

    # ── Resume: skip if model already exists ──────────────────────────
    if not retrain and os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        return {
            "ticker": ticker,
            "status": "SKIPPED",
            "reason": "already trained",
            "size_mb": f"{size_mb:.1f}",
        }

    # ── Download with retry ───────────────────────────────────────────
    data = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            data = yf.download(ticker, period="max", progress=False, timeout=30)
            if not data.empty:
                break
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                return {"ticker": ticker, "status": "FAILED", "error": f"Download error: {e}"}

    if data is None or data.empty:
        return {"ticker": ticker, "status": "FAILED", "error": "No data returned"}

    # ── Fix MultiIndex ────────────────────────────────────────────────
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Remove duplicate columns if any
    data = data.loc[:, ~data.columns.duplicated()]

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    if not required_cols.issubset(set(data.columns)):
        return {"ticker": ticker, "status": "FAILED", "error": f"Missing columns: {required_cols - set(data.columns)}"}

    data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
    data.dropna(inplace=True)

    if len(data) < MIN_ROWS:
        return {"ticker": ticker, "status": "FAILED", "error": f"Only {len(data)} rows (need {MIN_ROWS})"}

    # ── Feature engineering ───────────────────────────────────────────
    try:
        data = add_features(data)
        data = add_target(data)
        data.dropna(inplace=True)
    except Exception as e:
        return {"ticker": ticker, "status": "FAILED", "error": f"Feature engineering: {e}"}

    if len(data) < MIN_ROWS_AFTER:
        return {"ticker": ticker, "status": "FAILED", "error": f"Only {len(data)} rows after features"}

    # ── Train / test split ────────────────────────────────────────────
    split_idx = int(len(data) * 0.8)
    train = data.iloc[:split_idx]
    test  = data.iloc[split_idx:]

    # ── Model training ────────────────────────────────────────────────
    try:
        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_split=100,
            max_depth=15,
            random_state=42,
            n_jobs=1,           # 1 here — parallelism is at the process level
            class_weight="balanced",
        )
        model.fit(train[PREDICTORS], train["Target"])
    except Exception as e:
        return {"ticker": ticker, "status": "FAILED", "error": f"Training: {e}"}

    # ── Evaluate ──────────────────────────────────────────────────────
    preds  = model.predict(test[PREDICTORS])
    proba  = model.predict_proba(test[PREDICTORS])
    thresh = (proba[:, 1] >= 0.55).astype(int)

    acc       = accuracy_score(test["Target"], preds)
    prec_def  = precision_score(test["Target"], preds,   zero_division=0)
    prec_thr  = precision_score(test["Target"], thresh,  zero_division=0)

    # ── Save ──────────────────────────────────────────────────────────
    try:
        joblib.dump(model, model_path)
    except Exception as e:
        return {"ticker": ticker, "status": "FAILED", "error": f"Save: {e}"}

    size_mb = os.path.getsize(model_path) / (1024 * 1024)

    # Save metadata for freshness tracking
    set_model_metadata(
        ticker=ticker,
        accuracy=acc,
        precision=prec_thr,
        data_start=str(data.index[0].date()),
        data_end=str(data.index[-1].date()),
        rows=len(data),
    )

    return {
        "ticker":    ticker,
        "status":    "OK",
        "rows":      len(data),
        "train_rows": len(train),
        "test_rows":  len(test),
        "accuracy":  f"{acc:.4f}",
        "prec_default": f"{prec_def:.4f}",
        "precision": f"{prec_thr:.4f}",
        "size_mb":   f"{size_mb:.1f}",
        "date_start": str(data.index[0].date()),
        "date_end":   str(data.index[-1].date()),
    }


# ─────────────────────────────────────────────────────────────────────
# CSV progress logger
# ─────────────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "ticker", "status", "rows", "train_rows", "test_rows",
    "accuracy", "prec_default", "precision", "size_mb",
    "date_start", "date_end", "error", "reason",
]

def append_csv(path: str, row: dict):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train 500 Indian stock ML models")
    parser.add_argument("--retrain",      action="store_true",   help="Force retrain even if model exists")
    parser.add_argument("--workers",      type=int, default=DEFAULT_WORKERS, help="Parallel worker processes")
    parser.add_argument("--tickers-only", action="store_true",   help="Print ticker list and exit")
    args = parser.parse_args()

    if args.tickers_only:
        for i, t in enumerate(ALL_500_TICKERS, 1):
            print(f"{i:>4}. {t}")
        print(f"\nTotal: {len(ALL_500_TICKERS)} tickers")
        return

    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)

    log_csv = os.path.join(models_dir, "training_log.csv")

    print("=" * 65)
    print("  TOP 500 INDIAN STOCKS — Model Training Pipeline")
    print("=" * 65)
    print(f"  Models directory : {models_dir}")
    print(f"  Stocks to train  : {len(ALL_500_TICKERS)}")
    print(f"  Parallel workers : {args.workers}")
    print(f"  Retrain mode     : {'YES — overwriting existing' if args.retrain else 'NO — skipping done'}")
    print(f"  Progress log     : {log_csv}")
    print("=" * 65 + "\n")

    tasks = [(t, models_dir, args.retrain) for t in ALL_500_TICKERS]

    results   = []
    ok_count  = 0
    fail_count= 0
    skip_count= 0
    start     = time.time()
    done      = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(train_single_stock, task): task[0] for task in tasks}

        for future in as_completed(futures):
            done += 1
            ticker = futures[future]

            try:
                result = future.result()
            except Exception as exc:
                result = {"ticker": ticker, "status": "FAILED", "error": str(exc)}
                traceback.print_exc()

            results.append(result)
            append_csv(log_csv, result)

            status = result.get("status", "?")
            elapsed = time.time() - start
            eta = (elapsed / done) * (len(tasks) - done) if done else 0

            if status == "OK":
                ok_count += 1
                tag = "[OK]     "
                detail = f"acc={result.get('accuracy','')} prec={result.get('precision','')} ({result.get('size_mb','')} MB)"
            elif status == "SKIPPED":
                skip_count += 1
                tag = "[SKIP]   "
                detail = result.get("reason", "")
            else:
                fail_count += 1
                tag = "[FAILED] "
                detail = result.get("error", "")

            print(
                f"[{done:>3}/{len(tasks)}] {tag} {ticker:<22}  {detail}"
                f"   |  ETA {eta/60:.1f}m"
            )

    elapsed_total = time.time() - start

    # ── Final summary ─────────────────────────────────────────────────
    print("\n\n" + "=" * 65)
    print("  TRAINING COMPLETE — SUMMARY")
    print("=" * 65)
    print(f"  Successful : {ok_count}")
    print(f"  Skipped    : {skip_count}  (already trained)")
    print(f"  Failed     : {fail_count}")
    print(f"  Total time : {elapsed_total:.0f}s  ({elapsed_total/60:.1f} min)")
    print(f"  Log saved  : {log_csv}")

    ok_results = [r for r in results if r.get("status") == "OK"]
    if ok_results:
        accs  = [float(r["accuracy"])  for r in ok_results if r.get("accuracy")]
        precs = [float(r["precision"]) for r in ok_results if r.get("precision")]
        print(f"\n  Avg accuracy  : {np.mean(accs):.2%}")
        print(f"  Avg precision : {np.mean(precs):.2%}")

        total_size = sum(float(r.get("size_mb", 0)) for r in ok_results)
        print(f"  Total model disk usage: {total_size:.0f} MB  ({total_size/1024:.2f} GB)")

    failed = [r for r in results if r.get("status") == "FAILED"]
    if failed:
        print(f"\n  Failed tickers:")
        for r in failed:
            print(f"    [X] {r['ticker']:<22}  {r.get('error','')}")

    print(f"\n{'='*65}")
    print(f"  Models saved to: {models_dir}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
