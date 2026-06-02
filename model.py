"""
model.py — Dynamic multi-stock prediction engine for NIFTY 50.
Models are loaded on demand and cached in memory.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import os

from model_metadata import (
    get_model_age_days,
    get_model_metadata,
    is_model_stale,
    MODEL_MAX_AGE_DAYS,
)

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
    "ADANIGREEN.NS": "Adani Green Energy",
    "ADANITRANS.NS": "Adani Transmission",
    "AMBUJACEM.NS": "Ambuja Cements",
    "AUROPHARMA.NS": "Aurobindo Pharma",
    "BAJAJHLDNG.NS": "Bajaj Holdings",
    "BANKBARODA.NS": "Bank of Baroda",
    "BEL.NS": "Bharat Electronics",
    "BERGEPAINT.NS": "Berger Paints",
    "BIOCON.NS": "Biocon",
    "BOSCHLTD.NS": "Bosch",
    "CANBK.NS": "Canara Bank",
    "CHOLAFIN.NS": "Chola Finance",
    "COLPAL.NS": "Colgate-Palmolive",
    "DABUR.NS": "Dabur",
    "DLF.NS": "DLF",
    "GAIL.NS": "GAIL",
    "GODREJCP.NS": "Godrej Consumer",
    "HAVELLS.NS": "Havells",
    "ICICIPRULI.NS": "ICICI Prudential",
    "ICICIGI.NS": "ICICI General Insurance",
    "INDUSTOWER.NS": "Indus Towers",
    "IOC.NS": "Indian Oil",
    "IRCTC.NS": "IRCTC",
    "JINDALSTEL.NS": "Jindal Steel",
    "LUPIN.NS": "Lupin",
    "MCDOWELL-N.NS": "McDowell",
    "MUTHOOTFIN.NS": "Muthoot Finance",
    "NAUKRI.NS": "Naukri",
    "NMDC.NS": "NMDC",
    "PAGEIND.NS": "Page Industries",
    "PERSISTENT.NS": "Persistent Systems",
    "PETRONET.NS": "Petronet LNG",
    "PIIND.NS": "PI Industries",
    "PIDILITIND.NS": "Pidilite",
    "PNB.NS": "Punjab National Bank",
    "POLYCAB.NS": "Polycab",
    "RECLTD.NS": "REC",
    "SAIL.NS": "SAIL",
    "SIEMENS.NS": "Siemens",
    "SRF.NS": "SRF",
    "TORNTPHARM.NS": "Torrent Pharma",
    "TRENT.NS": "Trent",
    "TVSMOTOR.NS": "TVS Motor",
    "UBL.NS": "UBL",
    "UNITDSPR.NS": "United Spirits",
    "UPL.NS": "UPL",
    "VEDL.NS": "Vedanta",
    "VOLTAS.NS": "Voltas",
    "ZYDUSLIFE.NS": "Zydus Life",
    "HAL.NS": "HAL",
}

# Top 500 stocks: NIFTY 50 + NIFTY Next 50 + NIFTY Midcap 150 + NIFTY Smallcap 250
ALL_500_STOCKS_TICKERS = [
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
    "VBL.NS", "WHIRLPOOL.NS", "WOCKPHARMA.NS", "ZEEL.NS",
    "ZENTEC.NS", "ZENSARTECH.NS", "3MINDIA.NS", "AARTIDRUGS.NS", "ABBOTINDIA.NS",
    "AEGISCHEM.NS", "AFFLE.NS", "AJANTPHARM.NS", "ALKEM.NS", "AMBER.NS",
    "ANGELONE.NS", "APOLLOTYRE.NS", "APTUS.NS", "AVANTIFEED.NS",
    "BASF.NS", "BAYERCROP.NS", "BBTC.NS", "BLUEDART.NS", "BLUESTARCO.NS",
    "BRIGADE.NS", "BSE.NS", "CAMPUS.NS", "CARBORUNIV.NS", "CCL.NS",
    "CENTURYTEX.NS", "CHAMBLFERT.NS", "CHEMCON.NS", "CIGNITI.NS", "CLEAN.NS",
    "AAVAS.NS", "ACCELYA.NS", "ACE.NS", "ADANIPOWER.NS", "AETHER.NS",
    "AGIIL.NS", "AGROPHOS.NS", "AIIL.NS", "AKZOINDIA.NS", "ALANKIT.NS",
    "ALEMBICLTD.NS", "ALICON.NS", "ALKYLAMINE.NS", "ALLCARGO.NS", "AMARAJABAT.NS",
    "AMJLAND.NS", "ANANTRAJ.NS", "ANDHRSUGAR.NS", "ANGELBRKG.NS", "ANSALAPI.NS",
    "APARINDS.NS", "APOLLOPIPE.NS", "ARCOTECH.NS", "ARFIN.NS", "ARIHANTCAP.NS",
    "ARMANFIN.NS", "ARROWGREEN.NS", "ARVIND.NS", "ARVINDFASN.NS", "ASAHIINDIA.NS",
    "ASALCBR.NS", "ASHIANA.NS", "ASHIMASYN.NS", "ASKAUTOLTD.NS", "ASMTEC.NS",
    "ATGL.NS", "ATIL.NS", "AUROBINDO.NS", "AVTNPL.NS", "AXISGOLD.NS",
    "AZAD.NS", "BAJAJCON.NS", "BALAJITELE.NS", "BALMLAWRIE.NS", "BANARISUG.NS",
    "BANSWRAS.NS", "BARBEQUE.NS", "BASML.NS", "BDHL.NS", "BEML.NS",
    "BFUTILITIE.NS", "BHAGCHEM.NS", "BHANDARI.NS", "BHARATFORG.NS", "BHARATRAS.NS",
    "BIKAJI.NS", "BINDALAGRO.NS", "BIRLACORPN.NS", "BKMINDST.NS", "BOROLTD.NS",
    "BOROSIL.NS", "BQLIND.NS", "CRAFTSMAN.NS", "CREATIVSER.NS", "CRISIL.NS",
    "CYIENTDLM.NS", "DALMIASUG.NS", "DATAMATICS.NS", "DBCORP.NS", "DCMSHRIRAM.NS",
    "DCXSYS.NS", "DOMS.NS", "DREDGECORP.NS", "DUROPLY.NS", "DYNAMATECH.NS",
    "EASEMYTRIP.NS", "EFCLTD.NS", "EIDPARRY.NS", "ELECON.NS", "EMKAYTOOLS.NS",
    "ENDURANCE.NS", "ENIL.NS", "EPIGRAL.NS", "EQUITASBNK.NS", "ESSARSHPNG.NS",
    "ETHOSLTD.NS", "EUROBATIND.NS", "EXLSERVICE.NS", "FAIRCHEM.NS", "FCSSOFT.NS",
    "FINEORG.NS", "FINOLEX.NS", "FINOPB.NS", "FLFL.NS", "FOODWORKS.NS",
    "FORCE.NS", "GANDHAR.NS", "GARFIBRES.NS", "GARNET.NS", "GEOJITFSL.NS",
    "GHCL.NS", "GLOBUSSPR.NS", "GLODYNE.NS", "GNFC.NS", "GPIL.NS",
    "GPTINFRA.NS", "GRAPHITE.NS", "GRAVITA.NS", "GREENPOWER.NS", "GREENPANEL.NS",
    "GRINFRA.NS", "GRSE.NS", "GTLINFRA.NS", "GUFICBIO.NS", "GULFOILLUB.NS",
    "HARDWYN.NS", "HATHWAY.NS", "HEMIPROP.NS", "HERITGFOOD.NS", "HIKAL.NS",
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
    "OPTIEMUS.NS", "ORISSAMINE.NS", "OSIAJEE.NS", "PALREDTEC.NS", "PANAMAPET.NS",
    "PATELENG.NS", "PATINTLOG.NS", "PCJEWELLER.NS", "PDSL.NS", "PENIND.NS",
    "PGINDUSTRY.NS", "PGIL.NS", "PILANIINVS.NS", "PLASTIBLEN.NS", "POKARNA.NS",
    "POLYMED.NS", "POONAWALLA.NS", "POWERMECH.NS", "PRAKASH.NS", "PRICOLLTD.NS",
    "PRINCEPIPE.NS", "PRISMJOINTS.NS", "PRIVISCL.NS", "PSPPROJECT.NS", "PTCIL.NS",
    "PURVA.NS", "QUESS.NS", "RADHIKJWE.NS", "RADICO.NS", "RAILTEL.NS",
    "RAJRATAN.NS", "RAJSREESUG.NS", "RALLIS.NS", "RAMCOIND.NS", "RATNAMANI.NS",
    "RAYMOND.NS", "RBA.NS", "RBLBANK.NS", "REDINGTON.NS", "REPCO.NS",
    "RESPONIND.NS", "RKFORGE.NS", "ROHLTD.NS", "RUPA.NS", "RUSHIL.NS",
    "SAFARI.NS", "SALZERELEC.NS", "SANOFI.NS", "SAPPHIRE.NS", "SARDA.NS",
    "SAREGAMA.NS", "SATIA.NS", "SBFC.NS", "SEPOWER.NS", "SHAKTIPUMP.NS",
    "SHANTIGEAR.NS", "SHARDACROP.NS", "SHAREINDIA.NS", "SHILPAMED.NS", "SHREDIGCEM.NS",
    "SHYAMMETL.NS", "SIGIND.NS", "SKPMAR.NS", "SMLISUZU.NS", "SNOWMAN.NS",
    "SOLARA.NS", "SONATSOFTW.NS", "SOUTHBANK.NS", "SPANDANA.NS", "SPECIALITY.NS",
    "SPTL.NS", "SRINDLTH.NS", "SSWL.NS", "STAR.NS", "STCINDIA.NS",
    "STEELXIND.NS", "STYRENIX.NS", "SUBROS.NS", "SUDARSCHEM.NS", "SUGIND.NS",
    "SUKHJITS.NS", "SURYAROSNI.NS", "SUULD.NS", "SWSOLAR.NS", "SYMPHONY.NS",
    "SYNCOMF.NS", "TARSONS.NS", "TATAMETALI.NS", "TCIEXP.NS", "TCNSBRANDS.NS",
    "TDPOWERSYS.NS", "TEXRAIL.NS", "THYROCARE.NS", "TINPLATE.NS", "TIPSINDLTD.NS",
    "TIRUMALCHM.NS", "TMRVL.NS", "TNPETRO.NS", "TPLPLASTEH.NS", "TREEHOUSE.NS",
    "TRIL.NS", "TRIVENI.NS",
]

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
    seen = set()
    for ticker in ALL_500_STOCKS_TICKERS:
        if ticker in seen:
            continue
        seen.add(ticker)
        short = ticker.replace(".NS", "")
        name = NIFTY_50_STOCKS.get(ticker, short)
        stocks.append({"ticker": ticker, "short": short, "name": name,
                        "label": f"{short} — {name}"})
    return stocks


def get_stock_predictions(stocks):
    """Get predictions for all stocks and sort by bullish/bearish."""
    predictions = []

    for stock in stocks:
        try:
            result = predict_stock(stock['ticker'])
            if result and isinstance(result, dict) and "error" not in result:
                stock_copy = stock.copy()
                stock_copy['prediction'] = result.get('trend', 'Unknown')
                stock_copy['confidence'] = result.get('confidence', '0%')
                predictions.append(stock_copy)
        except:
            pass

    # Sort by bullish first, then by confidence
    bullish = [s for s in predictions if s['prediction'] == 'Bullish']
    bearish = [s for s in predictions if s['prediction'] == 'Bearish']
    bullish.sort(key=lambda x: float(x['confidence'].strip('%')) / 100, reverse=True)
    bearish.sort(key=lambda x: float(x['confidence'].strip('%')) / 100, reverse=True)

    return bullish + bearish


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

        # Model freshness info
        model_age = get_model_age_days(stock)
        model_meta = get_model_metadata(stock)
        model_stale = is_model_stale(stock)
        last_trained_str = ""
        if model_meta and "last_trained" in model_meta:
            try:
                from datetime import datetime
                trained_dt = datetime.fromisoformat(model_meta["last_trained"])
                last_trained_str = trained_dt.strftime("%d %b %Y, %I:%M %p UTC")
            except (ValueError, TypeError):
                last_trained_str = model_meta.get("last_trained", "Unknown")

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
            # Model freshness fields
            "model_age_days": model_age,
            "last_trained": last_trained_str,
            "is_stale": model_stale,
            "max_age_days": MODEL_MAX_AGE_DAYS,
        }
    except Exception as e:
        print(f"DEBUG: Error in predict_stock: {e}")
        import traceback
        traceback.print_exc()
        return None