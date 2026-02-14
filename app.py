"""
SignalScope — Stock Signal Scanner
Scans NIFTY 50, NIFTY 100, BSE 100, and NIFTY Midcap 150.
Each stock is tagged with its index memberships.
"""

import os
import time
import threading
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyotp
import requests
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from SmartApi import SmartConnect
from index_data import build_index_map, get_all_symbols, NIFTY_50, NIFTY_NEXT_50, BSE_100, NIFTY_MIDCAP_150

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

API_KEY    = os.getenv("ANGEL_API_KEY", "")
CLIENT_ID  = os.getenv("ANGEL_CLIENT_ID", "")
PASSWORD   = os.getenv("ANGEL_PASSWORD", "")
TOTP_TOKEN = os.getenv("ANGEL_TOTP_TOKEN", "")

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "900"))

# Build index membership map
INDEX_MAP = build_index_map()
ALL_SYMBOLS = get_all_symbols()

smart_api = None
refresh_token = None
instrument_list = []
is_scanning = False
credentials_ok = False

scan_results = {
    "last_scan": None, "status": "not_started", "total_scanned": 0,
    "buy_signals": [], "sell_signals": [], "all_stocks": [],
    "portfolio_holdings": [], "errors": [],
    "index_counts": {"NIFTY 50": 0, "NIFTY 100": 0, "BSE 100": 0, "MIDCAP 150": 0},
}

# ═══════════════════════════════════════════════════════════════
# AUTH
# ═══════════════════════════════════════════════════════════════

def check_credentials():
    missing = []
    if not API_KEY: missing.append("ANGEL_API_KEY")
    if not CLIENT_ID: missing.append("ANGEL_CLIENT_ID")
    if not PASSWORD: missing.append("ANGEL_PASSWORD")
    if not TOTP_TOKEN: missing.append("ANGEL_TOTP_TOKEN")
    return missing

def create_session():
    global smart_api, refresh_token, credentials_ok
    try:
        smart_api = SmartConnect(api_key=API_KEY)
        totp = pyotp.TOTP(TOTP_TOKEN).now()
        data = smart_api.generateSession(CLIENT_ID, PASSWORD, totp)
        if not data or data.get("status") is False:
            logger.error(f"Login failed: {data}")
            credentials_ok = False
            return False
        refresh_token = data["data"]["refreshToken"]
        credentials_ok = True
        logger.info("Logged in to Angel One")
        return True
    except Exception as e:
        logger.error(f"Login error: {e}")
        credentials_ok = False
        return False

def ensure_session():
    if smart_api is None: return create_session()
    try:
        smart_api.getProfile(refresh_token)
        return True
    except:
        return create_session()

# ═══════════════════════════════════════════════════════════════
# INSTRUMENTS
# ═══════════════════════════════════════════════════════════════

def fetch_instrument_list():
    global instrument_list
    try:
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        logger.info("Downloading instrument list...")
        resp = requests.get(url, timeout=60)
        all_inst = resp.json()
        eq = [i for i in all_inst if i.get("exch_seg") == "NSE" and i.get("symbol", "").endswith("-EQ")]

        # Filter to only stocks in our indices
        instrument_list = [i for i in eq if i["symbol"].replace("-EQ", "") in ALL_SYMBOLS]

        logger.info(f"Loaded {len(instrument_list)} stocks across all indices (target: {len(ALL_SYMBOLS)})")
        return True
    except Exception as e:
        logger.error(f"Instrument fetch failed: {e}")
        return False

# ═══════════════════════════════════════════════════════════════
# DATA FETCH
# ═══════════════════════════════════════════════════════════════

def fetch_candle_data(symbol_token):
    """
    Fetch daily candles. 365 days gives ~250 trading days, enough for SMA(200).
    SMA(200) now means 200 DAYS (~10 months), RSI = daily momentum, etc.
    """
    if not ensure_session(): return None
    try:
        to_d = datetime.now()
        from_d = to_d - timedelta(days=365)
        resp = smart_api.getCandleData({
            "exchange": "NSE", "symboltoken": str(symbol_token),
            "interval": "ONE_DAY",
            "fromdate": from_d.strftime("%Y-%m-%d %H:%M"),
            "todate": to_d.strftime("%Y-%m-%d %H:%M"),
        })
        if resp and resp.get("status") and resp.get("data"):
            df = pd.DataFrame(resp["data"], columns=["DateTime","Open","High","Low","Close","Volume"])
            df["DateTime"] = pd.to_datetime(df["DateTime"])
            df.set_index("DateTime", inplace=True)
            return df
        return None
    except Exception as e:
        logger.debug(f"Candle error {symbol_token}: {e}")
        return None

def fetch_portfolio():
    if not ensure_session(): return []
    try:
        result = smart_api.holding()
        if result and result.get("status") and result.get("data"):
            return [{"symbol": h.get("tradingsymbol",""), "token": h.get("symboltoken",""),
                      "quantity": h.get("quantity",0), "avg_price": float(h.get("averageprice",0)),
                      "ltp": float(h.get("ltp",0)), "pnl": float(h.get("profitandloss",0))}
                     for h in result["data"]]
        return []
    except Exception as e:
        logger.error(f"Portfolio error: {e}")
        return []

# ═══════════════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════════════

def calc_sma(s, p): return s.rolling(window=p, min_periods=p).mean()

def calc_rsi(s, p=14):
    d = s.diff(); g = d.where(d>0,0.0); l = -d.where(d<0,0.0)
    ag = g.ewm(com=p-1,min_periods=p).mean(); al = l.ewm(com=p-1,min_periods=p).mean()
    return 100 - (100/(1+ag/al))

def calc_bb(s, p=20, sd=2):
    m = calc_sma(s,p); st = s.rolling(window=p,min_periods=p).std()
    return m, m+sd*st, m-sd*st

def calc_macd(s, f=12, sl=26, sg=9):
    ef = s.ewm(span=f,adjust=False).mean(); es = s.ewm(span=sl,adjust=False).mean()
    ml = ef-es; sig = ml.ewm(span=sg,adjust=False).mean()
    return ml, sig, ml-sig

def calc_obv(close, vol):
    r = pd.Series(index=close.index, dtype=float); r.iloc[0] = 0
    for i in range(1,len(close)):
        if close.iloc[i]>close.iloc[i-1]: r.iloc[i]=r.iloc[i-1]+vol.iloc[i]
        elif close.iloc[i]<close.iloc[i-1]: r.iloc[i]=r.iloc[i-1]-vol.iloc[i]
        else: r.iloc[i]=r.iloc[i-1]
    return r

def analyze_stock(df):
    if df is None or len(df) < 200: return None
    close, vol = df["Close"], df["Volume"]
    price = close.iloc[-1]

    s200 = calc_sma(close,200).iloc[-1]
    r14 = calc_rsi(close,14).iloc[-1]
    bm,bu,bl = calc_bb(close,20,2)
    ml,ms,mh = calc_macd(close)
    ov = calc_obv(close,vol)

    cbm,cbu,cbl = bm.iloc[-1],bu.iloc[-1],bl.iloc[-1]
    cm,cms,cmh = ml.iloc[-1],ms.iloc[-1],mh.iloc[-1]
    co = ov.iloc[-1]
    co5 = ov.iloc[-5] if len(ov)>=5 else ov.iloc[0]
    co20 = ov.iloc[-20] if len(ov)>=20 else ov.iloc[0]
    tlb = any(close.iloc[i]<=bl.iloc[i] for i in range(-5,0) if not pd.isna(bl.iloc[i]))

    ok = lambda v: not pd.isna(v)
    sf = lambda v,d=2: round(float(v),d) if ok(v) else None

    buy_c = {
        "trend_filter": bool(price>s200) if ok(s200) else False,
        "momentum_filter": bool(30<r14<=45) if ok(r14) else False,
        "volatility_filter": bool(price<=cbm or tlb) if ok(cbm) else False,
        "momentum_confirm": bool(cm>cms and cmh>0) if ok(cm) and ok(cms) else False,
        "volume_confirm": bool(co>co5 and co>co20),
    }
    sell_c = {
        "trend_break": bool(price<s200) if ok(s200) else False,
        "momentum_reversal": bool(r14>=65) if ok(r14) else False,
        "volatility_extreme": bool(price>=cbu) if ok(cbu) else False,
        "momentum_fade": bool(cm<cms and cmh<0) if ok(cm) and ok(cms) else False,
        "volume_weakness": bool(co<co5),
    }

    bc,sc = sum(buy_c.values()),sum(sell_c.values())
    return {
        "price":sf(price),"sma200":sf(s200),"rsi":sf(r14),
        "bb_upper":sf(cbu),"bb_mid":sf(cbm),"bb_lower":sf(cbl),
        "macd":sf(cm,4),"macd_signal":sf(cms,4),"macd_hist":sf(cmh,4),
        "obv":sf(co,0),
        "buy_conditions":buy_c,"sell_conditions":sell_c,
        "buy_count":f"{bc}/5","sell_count":f"{sc}/5",
        "is_buy":bc==5,"is_sell":sc>=3,
        "buy_pct":bc/5*100,"sell_pct":sc/5*100,
    }

# ═══════════════════════════════════════════════════════════════
# SCAN
# ═══════════════════════════════════════════════════════════════

def run_full_scan():
    global scan_results, is_scanning
    if not credentials_ok:
        logger.error("Cannot scan — not logged in")
        return
    if is_scanning: return
    is_scanning = True
    scan_results["status"] = "scanning"

    logger.info(f"Starting scan of {len(instrument_list)} stocks across all indices...")
    t0 = time.time()
    buys, sells, all_s, errs = [], [], [], []

    portfolio = fetch_portfolio()
    ptokens = {h["token"] for h in portfolio}

    # Track per-index counts
    idx_counts = {"NIFTY 50": 0, "NIFTY 100": 0, "BSE 100": 0, "MIDCAP 150": 0}

    for idx, inst in enumerate(instrument_list):
        sym = inst["symbol"]
        token = inst["token"]
        name = inst.get("name", sym.replace("-EQ",""))
        clean_sym = sym.replace("-EQ", "")

        try:
            df = fetch_candle_data(token)
            if df is None or len(df) < 200:
                errs.append({"symbol": sym, "error": "Not enough data"})
                continue

            analysis = analyze_stock(df)
            if analysis is None: continue

            # Get index memberships for this stock
            indices = INDEX_MAP.get(clean_sym, [])

            stock = {
                "symbol": sym, "name": name, "token": token,
                "in_portfolio": token in ptokens,
                "indices": indices,  # e.g. ["NIFTY 50", "NIFTY 100", "BSE 100"]
                **analysis,
            }
            all_s.append(stock)

            # Count per index
            for ix in indices:
                if ix in idx_counts:
                    idx_counts[ix] += 1

            if analysis["is_buy"]: buys.append(stock)
            if analysis["is_sell"] and token in ptokens: sells.append(stock)
            time.sleep(0.35)
        except Exception as e:
            errs.append({"symbol": sym, "error": str(e)})
        if (idx+1) % 25 == 0:
            logger.info(f"  Progress: {idx+1}/{len(instrument_list)}...")

    elapsed = time.time() - t0
    scan_results = {
        "last_scan": datetime.now().isoformat(), "status": "complete",
        "scan_duration_sec": round(elapsed,1), "total_scanned": len(all_s),
        "buy_signals": sorted(buys, key=lambda x: x["buy_pct"], reverse=True),
        "sell_signals": sorted(sells, key=lambda x: x["sell_pct"], reverse=True),
        "all_stocks": sorted(all_s, key=lambda x: x["buy_pct"], reverse=True),
        "portfolio_holdings": portfolio, "errors": errs,
        "index_counts": idx_counts,
    }
    is_scanning = False
    logger.info(f"Scan done in {elapsed:.0f}s — {len(all_s)} stocks, {len(buys)} buys, {len(sells)} sells")
    for ix, cnt in idx_counts.items():
        logger.info(f"  {ix}: {cnt} stocks scanned")

# ═══════════════════════════════════════════════════════════════
# SCHEDULER
# ═══════════════════════════════════════════════════════════════

def scheduler():
    while True:
        now = datetime.now()
        wd = now.weekday() < 5
        ao = (now.hour==9 and now.minute>=15) or now.hour>=10
        bc = now.hour<15 or (now.hour==15 and now.minute<=30)
        if wd and ao and bc:
            run_full_scan()
        else:
            logger.info("Outside market hours")
        time.sleep(SCAN_INTERVAL)

# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route("/api/status")
def api_status():
    return jsonify({
        "connected": credentials_ok, "instruments": len(instrument_list),
        "last_scan": scan_results.get("last_scan"),
        "scan_status": scan_results.get("status"),
        "is_scanning": is_scanning,
        "index_counts": scan_results.get("index_counts", {}),
        "missing": check_credentials(),
    })

@app.route("/api/scan", methods=["POST"])
def api_trigger_scan():
    if not credentials_ok: return jsonify({"error": "Not connected"}), 401
    if is_scanning: return jsonify({"status": "already_scanning"}), 409
    threading.Thread(target=run_full_scan, daemon=True).start()
    return jsonify({"status": "scan_started"})

@app.route("/api/results")
def api_results():
    return jsonify(scan_results)

@app.route("/api/stock/<symbol>")
def api_stock_detail(symbol):
    for s in scan_results.get("all_stocks", []):
        if s["symbol"] == symbol or s["name"] == symbol: return jsonify(s)
    return jsonify({"error": "Not found"}), 404

@app.route("/")
def serve_frontend():
    return send_from_directory(".", "index.html")

# ═══════════════════════════════════════════════════════════════
# INIT
# ═══════════════════════════════════════════════════════════════

def initialize():
    global credentials_ok
    missing = check_credentials()
    if missing:
        logger.error(f"Missing: {', '.join(missing)}")
        credentials_ok = False
        return False
    if not create_session():
        logger.error("Login failed!")
        return False
    fetch_instrument_list()
    threading.Thread(target=scheduler, daemon=True).start()
    threading.Thread(target=run_full_scan, daemon=True).start()
    return True