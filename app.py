"""
SignalScope — Stock Signal Scanner (v2 — Weighted Scoring)
Daily candles, 6 indicators, 100-point buy scoring system.
Support/Resistance with Risk:Reward ratio.
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
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from SmartApi import SmartConnect
from index_data import build_index_map, get_all_symbols

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

API_KEY    = os.getenv("ANGEL_API_KEY", "")
CLIENT_ID  = os.getenv("ANGEL_CLIENT_ID", "")
PASSWORD   = os.getenv("ANGEL_PASSWORD", "")
TOTP_TOKEN = os.getenv("ANGEL_TOTP_TOKEN", "")
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "900"))

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
        instrument_list = [i for i in eq if i["symbol"].replace("-EQ", "") in ALL_SYMBOLS]
        logger.info(f"Loaded {len(instrument_list)} stocks (target: {len(ALL_SYMBOLS)})")
        return True
    except Exception as e:
        logger.error(f"Instrument fetch failed: {e}")
        return False

# ═══════════════════════════════════════════════════════════════
# DATA FETCH
# ═══════════════════════════════════════════════════════════════

def fetch_candle_data(symbol_token):
    """Fetch 365 days of daily candles. ~250 trading days for SMA(200)."""
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
# TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════

def calc_sma(s, p):
    return s.rolling(window=p, min_periods=p).mean()

def calc_rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0.0)
    l = -d.where(d < 0, 0.0)
    ag = g.ewm(com=p-1, min_periods=p).mean()
    al = l.ewm(com=p-1, min_periods=p).mean()
    return 100 - (100 / (1 + ag / al))

def calc_bb(s, p=20, sd=2):
    m = calc_sma(s, p)
    st = s.rolling(window=p, min_periods=p).std()
    return m, m + sd * st, m - sd * st

def calc_macd(s, f=12, sl=26, sg=9):
    ef = s.ewm(span=f, adjust=False).mean()
    es = s.ewm(span=sl, adjust=False).mean()
    ml = ef - es
    sig = ml.ewm(span=sg, adjust=False).mean()
    return ml, sig, ml - sig

def calc_obv(close, vol):
    r = pd.Series(index=close.index, dtype=float)
    r.iloc[0] = 0
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            r.iloc[i] = r.iloc[i-1] + vol.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            r.iloc[i] = r.iloc[i-1] - vol.iloc[i]
        else:
            r.iloc[i] = r.iloc[i-1]
    return r

def calc_adx(high, low, close, p=14):
    """Average Directional Index — measures trend strength (0-100)."""
    plus_dm = high.diff()
    minus_dm = low.diff().abs() * -1  # make positive for comparison

    # Correct +DM / -DM logic
    pdf = pd.DataFrame({"ph": high.diff(), "pl": low.shift(1) - low})
    pdf["plus_dm"] = pdf.apply(lambda r: r["ph"] if r["ph"] > r["pl"] and r["ph"] > 0 else 0, axis=1)
    pdf["minus_dm"] = pdf.apply(lambda r: r["pl"] if r["pl"] > r["ph"] and r["pl"] > 0 else 0, axis=1)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(span=p, adjust=False).mean()
    plus_di = 100 * (pdf["plus_dm"].ewm(span=p, adjust=False).mean() / atr)
    minus_di = 100 * (pdf["minus_dm"].ewm(span=p, adjust=False).mean() / atr)

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = dx.ewm(span=p, adjust=False).mean()
    return adx


def find_support_resistance(high, low, close, lookback=60):
    """
    Find nearest support and resistance levels using pivot points
    and recent swing highs/lows.
    """
    price = close.iloc[-1]
    recent_high = high.iloc[-lookback:]
    recent_low = low.iloc[-lookback:]
    recent_close = close.iloc[-lookback:]

    # Method 1: Swing highs and lows (local extrema over 5-day windows)
    swing_highs = []
    swing_lows = []
    for i in range(2, len(recent_high) - 2):
        if (recent_high.iloc[i] > recent_high.iloc[i-1] and
            recent_high.iloc[i] > recent_high.iloc[i-2] and
            recent_high.iloc[i] > recent_high.iloc[i+1] and
            recent_high.iloc[i] >= recent_high.iloc[i+2]):
            swing_highs.append(recent_high.iloc[i])
        if (recent_low.iloc[i] < recent_low.iloc[i-1] and
            recent_low.iloc[i] < recent_low.iloc[i-2] and
            recent_low.iloc[i] < recent_low.iloc[i+1] and
            recent_low.iloc[i] <= recent_low.iloc[i+2]):
            swing_lows.append(recent_low.iloc[i])

    # Method 2: Classic pivot point
    pp_high = recent_high.iloc[-1]
    pp_low = recent_low.iloc[-1]
    pp_close = recent_close.iloc[-1]
    pivot = (pp_high + pp_low + pp_close) / 3
    r1 = 2 * pivot - pp_low
    s1 = 2 * pivot - pp_high
    r2 = pivot + (pp_high - pp_low)
    s2 = pivot - (pp_high - pp_low)

    # Combine all levels
    resistance_levels = sorted(set([r1, r2] + [h for h in swing_highs if h > price]))
    support_levels = sorted(set([s1, s2] + [l for l in swing_lows if l < price]), reverse=True)

    # Nearest support (highest value below price)
    support = None
    for s in support_levels:
        if s < price:
            support = s
            break

    # Nearest resistance (lowest value above price)
    resistance = None
    for r in resistance_levels:
        if r > price:
            resistance = r
            break

    # Fallback
    if support is None:
        support = recent_low.min()
    if resistance is None:
        resistance = recent_high.max()

    return round(float(support), 2), round(float(resistance), 2)


# ═══════════════════════════════════════════════════════════════
# ANALYZE STOCK — WEIGHTED SCORING (100 POINTS)
# ═══════════════════════════════════════════════════════════════

def analyze_stock(df):
    """
    Weighted buy scoring (out of 100):
      SMA(200):  30 pts (binary)
      RSI(14):   20 pts + 5 bonus if RSI ≤ 35
      BB:        20 pts + 5 bonus if touched lower band
      MACD:      15 pts
      OBV:       10 pts
      ADX(14):   15 pts + 5 bonus if ADX > 30

    Sell scoring: same 5-condition system (3+/5 triggers sell)
    """
    if df is None or len(df) < 200:
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]
    price = close.iloc[-1]

    ok = lambda v: not pd.isna(v)
    sf = lambda v, d=2: round(float(v), d) if ok(v) else None

    # ── Compute indicators ──
    s200 = calc_sma(close, 200).iloc[-1]
    r14 = calc_rsi(close, 14).iloc[-1]
    bm, bu, bl = calc_bb(close, 20, 2)
    ml, ms, mh = calc_macd(close)
    ov = calc_obv(close, vol)
    adx_series = calc_adx(high, low, close, 14)

    cbm, cbu, cbl = bm.iloc[-1], bu.iloc[-1], bl.iloc[-1]
    cm, cms, cmh = ml.iloc[-1], ms.iloc[-1], mh.iloc[-1]
    # MACD slope: difference between last 2 MACD line values (positive = rising)
    cm_prev = ml.iloc[-2] if len(ml) >= 2 else cm
    macd_slope = float(cm - cm_prev) if ok(cm) and ok(cm_prev) else 0
    co = ov.iloc[-1]
    co5 = ov.iloc[-5] if len(ov) >= 5 else ov.iloc[0]
    co20 = ov.iloc[-20] if len(ov) >= 20 else ov.iloc[0]
    curr_adx = adx_series.iloc[-1] if len(adx_series) > 0 and ok(adx_series.iloc[-1]) else 0

    tlb = any(close.iloc[i] <= bl.iloc[i] for i in range(-5, 0) if ok(bl.iloc[i]))
    at_or_below_mid = bool(price <= cbm) if ok(cbm) else False

    # ── Support / Resistance ──
    support, resistance = find_support_resistance(high, low, close, 60)
    risk = round(price - support, 2) if support else 0
    reward = round(resistance - price, 2) if resistance else 0
    rr_ratio = round(reward / risk, 2) if risk > 0 else 0

    # ══════════════════════════════════════════════════════════
    # BUY SCORE (out of 100)
    # ══════════════════════════════════════════════════════════
    buy_score = 0
    buy_breakdown = {}

    # 1. SMA(200) — 30 pts (all or nothing)
    sma_pass = bool(price > s200) if ok(s200) else False
    sma_pts = 30 if sma_pass else 0
    buy_score += sma_pts
    buy_breakdown["sma200"] = {"pass": sma_pass, "pts": sma_pts, "max": 30,
                                "val": sf(s200), "desc": "Close > SMA(200)"}

    # 2. RSI(14) — 20 pts + 5 bonus
    rsi_pass = bool(30 < r14 <= 45) if ok(r14) else False
    rsi_deep = bool(r14 <= 35) if ok(r14) else False
    rsi_pts = 0
    if ok(r14) and rsi_pass:
        rsi_pts = 20
        if rsi_deep:
            rsi_pts += 5  # bonus for deeper dip
    buy_score += rsi_pts
    buy_breakdown["rsi"] = {"pass": rsi_pass, "pts": rsi_pts, "max": 25,
                             "val": sf(r14), "desc": "RSI 30-45" + (" +bonus ≤35" if rsi_deep else "")}

    # 3. Bollinger Bands — 20 pts + 5 bonus
    bb_pass = bool(at_or_below_mid or tlb) if ok(cbm) else False
    bb_pts = 0
    if bb_pass:
        bb_pts = 20
        if tlb:
            bb_pts += 5  # bonus for touching lower band
    buy_score += bb_pts
    buy_breakdown["bollinger"] = {"pass": bb_pass, "pts": bb_pts, "max": 25,
                                   "val": sf(cbm), "desc": "At/below mid BB" + (" +touched lower" if tlb else "")}

    # 4. MACD — 15 pts
    macd_pass = bool(cm > cms and cmh > 0) if ok(cm) and ok(cms) else False
    macd_pts = 15 if macd_pass else 0
    buy_score += macd_pts
    buy_breakdown["macd"] = {"pass": macd_pass, "pts": macd_pts, "max": 15,
                              "val": sf(cm, 4), "desc": "MACD > Signal, Hist > 0"}

    # 5. OBV — 10 pts
    obv_pass = bool(co > co5 and co > co20)
    obv_pts = 10 if obv_pass else 0
    buy_score += obv_pts
    buy_breakdown["obv"] = {"pass": obv_pass, "pts": obv_pts, "max": 10,
                             "val": sf(co, 0), "desc": "OBV rising vs 5d & 20d"}

    # 6. ADX(14) — 15 pts + 5 bonus
    adx_pass = bool(curr_adx > 25)
    adx_strong = bool(curr_adx > 30)
    adx_pts = 0
    if adx_pass:
        adx_pts = 15
        if adx_strong:
            adx_pts += 5  # bonus for very strong trend
    buy_score += adx_pts
    buy_breakdown["adx"] = {"pass": adx_pass, "pts": adx_pts, "max": 20,
                             "val": sf(curr_adx), "desc": "ADX > 25" + (" +bonus >30" if adx_strong else "")}

    # Buy signal classification
    if buy_score >= 80:
        buy_signal = "STRONG BUY"
    elif buy_score >= 70:
        buy_signal = "MODERATE BUY"
    else:
        buy_signal = "NO SIGNAL"

    # ══════════════════════════════════════════════════════════
    # SELL SCORE (old system: 3+ of 5 triggers)
    # ══════════════════════════════════════════════════════════
    sell_c = {
        "trend_break": bool(price < s200) if ok(s200) else False,
        "momentum_reversal": bool(r14 >= 65) if ok(r14) else False,
        "volatility_extreme": bool(price >= cbu) if ok(cbu) else False,
        "momentum_fade": bool(cm < cms and cmh < 0) if ok(cm) and ok(cms) else False,
        "volume_weakness": bool(co < co5),
    }
    sell_count = sum(sell_c.values())
    sell_pct = sell_count / 5 * 100

    return {
        "price": sf(price),
        "sma200": sf(s200),
        "rsi": sf(r14),
        "bb_upper": sf(cbu),
        "bb_mid": sf(cbm),
        "bb_lower": sf(cbl),
        "macd": sf(cm, 4),
        "macd_signal": sf(cms, 4),
        "macd_hist": sf(cmh, 4),
        "macd_slope": round(macd_slope, 4),
        "obv": sf(co, 0),
        "adx": sf(curr_adx),
        # Support / Resistance / Risk:Reward
        "support": support,
        "resistance": resistance,
        "risk": risk,
        "reward": reward,
        "rr_ratio": rr_ratio,
        # Buy scoring
        "buy_score": buy_score,
        "buy_signal": buy_signal,
        "buy_breakdown": buy_breakdown,
        # Sell scoring (kept simple)
        "sell_conditions": sell_c,
        "sell_count": f"{sell_count}/5",
        "sell_pct": sell_pct,
        "is_buy": buy_score >= 80,
        "is_moderate_buy": buy_score >= 70,
        "is_sell": sell_count >= 3,
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

    logger.info(f"Scanning {len(instrument_list)} stocks...")
    t0 = time.time()
    buys, sells, all_s, errs = [], [], [], []

    portfolio = fetch_portfolio()
    ptokens = {h["token"] for h in portfolio}
    idx_counts = {"NIFTY 50": 0, "NIFTY 100": 0, "BSE 100": 0, "MIDCAP 150": 0}

    for i, inst in enumerate(instrument_list):
        sym, token = inst["symbol"], inst["token"]
        name = inst.get("name", sym.replace("-EQ", ""))
        clean = sym.replace("-EQ", "")
        try:
            df = fetch_candle_data(token)
            if df is None or len(df) < 200:
                errs.append({"symbol": sym, "error": "Not enough data"})
                continue
            a = analyze_stock(df)
            if a is None: continue

            indices = INDEX_MAP.get(clean, [])
            stock = {"symbol": sym, "name": name, "token": token,
                     "in_portfolio": token in ptokens, "indices": indices, **a}
            all_s.append(stock)
            for ix in indices:
                if ix in idx_counts: idx_counts[ix] += 1
            if a["is_buy"] or a["is_moderate_buy"]: buys.append(stock)
            if a["is_sell"] and token in ptokens: sells.append(stock)
            time.sleep(0.35)
        except Exception as e:
            errs.append({"symbol": sym, "error": str(e)})
        if (i+1) % 25 == 0:
            logger.info(f"  {i+1}/{len(instrument_list)}...")

    elapsed = time.time() - t0
    scan_results = {
        "last_scan": datetime.now().isoformat(), "status": "complete",
        "scan_duration_sec": round(elapsed, 1), "total_scanned": len(all_s),
        "buy_signals": sorted(buys, key=lambda x: x["buy_score"], reverse=True),
        "sell_signals": sorted(sells, key=lambda x: x["sell_pct"], reverse=True),
        "all_stocks": sorted(all_s, key=lambda x: x["buy_score"], reverse=True),
        "portfolio_holdings": portfolio, "errors": errs, "index_counts": idx_counts,
    }
    is_scanning = False
    logger.info(f"Done {elapsed:.0f}s — {len(all_s)} stocks, {len(buys)} buys, {len(sells)} sells")

# ═══════════════════════════════════════════════════════════════
# SCHEDULER
# ═══════════════════════════════════════════════════════════════

def scheduler():
    while True:
        now = datetime.now()
        wd = now.weekday() < 5
        ao = (now.hour == 9 and now.minute >= 15) or now.hour >= 10
        bc = now.hour < 15 or (now.hour == 15 and now.minute <= 30)
        if wd and ao and bc: run_full_scan()
        else: logger.info("Outside market hours")
        time.sleep(SCAN_INTERVAL)

# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route("/api/status")
def api_status():
    return jsonify({"connected": credentials_ok, "instruments": len(instrument_list),
                    "last_scan": scan_results.get("last_scan"),
                    "scan_status": scan_results.get("status"), "is_scanning": is_scanning,
                    "index_counts": scan_results.get("index_counts", {}),
                    "missing": check_credentials()})

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

# ── Watchlist (stored server-side in memory, persists while app runs) ──
watchlists = {}  # { "list_name": ["SYMBOL-EQ", ...] }

@app.route("/api/watchlists")
def api_get_watchlists():
    return jsonify(watchlists)

@app.route("/api/watchlists/<name>", methods=["PUT"])
def api_create_watchlist(n):
    if n not in watchlists:
        watchlists[n] = []
    return jsonify({"status": "created", "n": n})

@app.route("/api/watchlists/<n>", methods=["DELETE"])
def api_delete_watchlist(n):
    watchlists.pop(n, None)
    return jsonify({"status": "deleted"})

@app.route("/api/watchlists/<n>/add", methods=["POST"])
def api_watchlist_add(n):
    symbol = request.json.get("symbol", "")
    if n not in watchlists:
        watchlists[n] = []
    if symbol and symbol not in watchlists[n]:
        watchlists[n].append(symbol)
    return jsonify({"status": "added", "watchlist": watchlists[n]})

@app.route("/api/watchlists/<n>/remove", methods=["POST"])
def api_watchlist_remove(n):
    symbol = request.json.get("symbol", "")
    if n in watchlists and symbol in watchlists[n]:
        watchlists[n].remove(symbol)
    return jsonify({"status": "removed", "watchlist": watchlists.get(n, [])})

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
