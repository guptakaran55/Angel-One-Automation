"""
SignalScope v4.0 — Multi-Market Edition
Scans NIFTY 500 (Angel One) + NASDAQ 100 (Yahoo Finance)
Daily candles, 6 indicators, weighted scoring, index charts.

v4.0 Changes:
  - Multi-market support: NIFTY 500 + NASDAQ 100
  - Index chart endpoint with 1D/1W/1M resolution
  - Parallel market scanning via ThreadPoolExecutor
  - yfinance integration for US market data
  - Market switcher in frontend
  - All v3.1 features preserved
"""

import os, time, threading, logging, hashlib, json
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd, pyotp, requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from SmartApi import SmartConnect
from index_data import get_scan_universe, build_index_tags, get_tags_for
from us_market import (
    get_nasdaq100_symbols, fetch_us_candle_data, fetch_index_chart_data,
    US_RATE_LIMITED
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

API_KEY    = os.getenv("ANGEL_API_KEY", "")
CLIENT_ID  = os.getenv("ANGEL_CLIENT_ID", "")
PASSWORD   = os.getenv("ANGEL_PASSWORD", "")
TOTP_TOKEN = os.getenv("ANGEL_TOTP_TOKEN", "")
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "900"))

MIN_AVG_VOLUME = int(os.getenv("MIN_AVG_VOLUME", "100000"))

APP_PASSWORD = os.getenv("APP_PASSWORD", "signal2026")

# Zerodha
ZERODHA_API_KEY    = os.getenv("ZERODHA_API_KEY", "")
ZERODHA_API_SECRET = os.getenv("ZERODHA_API_SECRET", "")
_zt = os.getenv("ZERODHA_ACCESS_TOKEN", "").strip()
zerodha_access_token = _zt if (_zt and not _zt.startswith("http") and len(_zt) > 20) else None
zerodha_user = None

INDEX_TAGS = build_index_tags()
SCAN_UNIVERSE = set()

smart_api = None
refresh_token = None
instrument_list = []
scan_instrument_list = []
is_scanning = {"nifty500": False, "nasdaq100": False}
abort_scan = {"nifty500": False, "nasdaq100": False}
credentials_ok = False
last_login_attempt = 0
login_backoff = 1

current_pace = 0.35
PACE_MIN = 0.35
PACE_MAX = 2.0
PACE_BACKOFF_MULT = 1.5
PACE_RECOVERY_MULT = 0.9

# ── Multi-market scan results ──
scan_progress = {
    "nifty500": {"current": 0, "total": 0, "ok": 0, "errors": 0, "skipped": 0, "rate_limited": False, "pace": 0.35},
    "nasdaq100": {"current": 0, "total": 0, "ok": 0, "errors": 0, "skipped": 0, "rate_limited": False, "pace": 0.5},
    "active_market": "nifty500",
}

scan_results = {
    "nifty500": {
        "last_scan": None, "status": "not_started", "total_scanned": 0,
        "buy_signals": [], "sell_signals": [], "all_stocks": [],
        "portfolio_holdings": [], "errors": [],
    },
    "nasdaq100": {
        "last_scan": None, "status": "not_started", "total_scanned": 0,
        "buy_signals": [], "sell_signals": [], "all_stocks": [],
        "portfolio_holdings": [], "errors": [],
    },
}

# ═══════════════════════════════════════════════════════════════
# AUTH (Angel One)
# ═══════════════════════════════════════════════════════════════

def check_credentials():
    missing = []
    if not API_KEY: missing.append("ANGEL_API_KEY")
    if not CLIENT_ID: missing.append("ANGEL_CLIENT_ID")
    if not PASSWORD: missing.append("ANGEL_PASSWORD")
    if not TOTP_TOKEN: missing.append("ANGEL_TOTP_TOKEN")
    return missing

def create_session():
    global smart_api, refresh_token, credentials_ok, last_login_attempt, login_backoff
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
        login_backoff = 1
        logger.info("Logged in to Angel One")
        return True
    except Exception as e:
        logger.error(f"Login error: {e}")
        credentials_ok = False
        return False

def ensure_session():
    global last_login_attempt, login_backoff, scan_progress
    if smart_api is None:
        return create_session()
    try:
        smart_api.getProfile(refresh_token)
        return True
    except:
        now = time.time()
        if now - last_login_attempt < login_backoff:
            return False
        last_login_attempt = now
        result = create_session()
        if not result:
            login_backoff = min(login_backoff * 2, 60)
            scan_progress["nifty500"]["rate_limited"] = True
            logger.warning(f"Login failed, backing off {login_backoff}s")
        else:
            login_backoff = 1
            scan_progress["nifty500"]["rate_limited"] = False
        return result

# ═══════════════════════════════════════════════════════════════
# INSTRUMENTS
# ═══════════════════════════════════════════════════════════════

def fetch_instrument_list():
    """Download NSE instruments from Angel One, filter to NIFTY 500 universe."""
    global instrument_list, scan_instrument_list, SCAN_UNIVERSE
    import urllib.request, ssl
    try:
        SCAN_UNIVERSE = get_scan_universe()
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        all_inst = None

        for attempt in range(1, 4):
            try:
                logger.info(f"Downloading Angel One instrument list (attempt {attempt}/3)...")
                ctx = ssl.create_default_context()
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/131.0.0.0",
                    "Accept": "application/json",
                })
                with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
                    raw = resp.read()
                logger.info(f"Downloaded {len(raw)/1024/1024:.1f} MB. Parsing JSON...")
                all_inst = json.loads(raw)
                logger.info(f"Parsed {len(all_inst)} instruments")
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt} failed: {e}")
                if attempt < 3:
                    time.sleep(5 * attempt)

        if not all_inst:
            logger.error("Could not download instrument list after 3 attempts.")
            logger.error("NIFTY scan unavailable. NASDAQ scan still works.")
            return False

        instrument_list = [
            i for i in all_inst
            if i.get("exch_seg") == "NSE" and i.get("symbol", "").endswith("-EQ")
        ]
        logger.info(f"Total NSE equities: {len(instrument_list)}")

        scan_instrument_list = [
            i for i in instrument_list
            if i["symbol"].replace("-EQ", "") in SCAN_UNIVERSE
        ]
        logger.info(f"Scan list: {len(scan_instrument_list)} stocks (NIFTY 500 universe)")
        return True
    except Exception as e:
        logger.error(f"Instrument fetch failed: {e}")
        return False

# ═══════════════════════════════════════════════════════════════
# DATA FETCH (Angel One)
# ═══════════════════════════════════════════════════════════════

RATE_LIMITED = "RATE_LIMITED"

def fetch_candle_data(symbol_token):
    if not ensure_session(): return RATE_LIMITED
    try:
        to_d = datetime.now()
        from_d = to_d - timedelta(days=730)
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
        err_str = str(e)
        if "access rate" in err_str.lower() or "rate" in err_str.lower():
            logger.warning(f"Rate limited on {symbol_token}")
            return RATE_LIMITED
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

def calc_sma(s, p): return s.rolling(window=p, min_periods=p).mean()

def calc_ema(s, p): return s.ewm(span=p, adjust=False).mean()

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
    direction = np.sign(close.diff()).fillna(0)
    return (direction * vol).cumsum()

def calc_adx(high, low, close, p=14):
    up_move = high.diff()
    down_move = low.shift(1) - low
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=p, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=p, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=p, adjust=False).mean() / atr)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = dx.ewm(span=p, adjust=False).mean()
    return adx, plus_di, minus_di

def find_support_resistance(high, low, close, lookback=60):
    price = close.iloc[-1]
    rh, rl, rc = high.iloc[-lookback:], low.iloc[-lookback:], close.iloc[-lookback:]
    window = 5
    swing_highs, swing_lows = [], []
    for i in range(window, len(rh) - window):
        is_high = True
        is_low = True
        for j in range(1, window + 1):
            if rh.iloc[i] <= rh.iloc[i - j] or rh.iloc[i] <= rh.iloc[i + j]:
                is_high = False
            if rl.iloc[i] >= rl.iloc[i - j] or rl.iloc[i] >= rl.iloc[i + j]:
                is_low = False
        if is_high: swing_highs.append(float(rh.iloc[i]))
        if is_low: swing_lows.append(float(rl.iloc[i]))

    def cluster_levels(levels, threshold_pct=1.5):
        if not levels: return []
        levels = sorted(levels)
        clusters = []
        current = [levels[0]]
        for lv in levels[1:]:
            if current and (lv - current[0]) / current[0] * 100 <= threshold_pct:
                current.append(lv)
            else:
                clusters.append(sum(current) / len(current))
                current = [lv]
        clusters.append(sum(current) / len(current))
        return clusters

    res_levels = cluster_levels([h for h in swing_highs if h > price])
    sup_levels = cluster_levels([l for l in swing_lows if l < price])

    pivot = (float(rh.iloc[-1]) + float(rl.iloc[-1]) + float(rc.iloc[-1])) / 3
    r1, s1 = 2 * pivot - float(rl.iloc[-1]), 2 * pivot - float(rh.iloc[-1])

    if not sup_levels: sup_levels = [s1] if s1 < price else [float(rl.min())]
    if not res_levels: res_levels = [r1] if r1 > price else [float(rh.max())]

    support = max([s for s in sup_levels if s < price], default=float(rl.min()))
    resistance = min([r for r in res_levels if r > price], default=float(rh.max()))
    return round(support, 2), round(resistance, 2)

# ═══════════════════════════════════════════════════════════════
# ANALYZE — WEIGHTED SCORING (works for both markets)
# ═══════════════════════════════════════════════════════════════

def analyze_stock(df, currency_symbol="₹", min_avg_volume=None):
    """Analyze a stock DataFrame. Works for any market — just pass the right currency."""
    if df is None or len(df) < 50: return None
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    n_candles = len(df)
    price = close.iloc[-1]
    ok = lambda v: not pd.isna(v)
    sf = lambda v, d=2: round(float(v),d) if ok(v) else None

    vol_threshold = min_avg_volume if min_avg_volume is not None else MIN_AVG_VOLUME
    avg_vol_20 = float(vol.iloc[-20:].mean()) if n_candles >= 20 else float(vol.mean())
    if vol_threshold > 0 and avg_vol_20 < vol_threshold:
        return None

    has_sma200 = n_candles >= 200
    has_sma50 = n_candles >= 50
    if has_sma200:
        s200 = calc_sma(close, 200).iloc[-1]
        s200_prev = calc_sma(close, 200).iloc[-2] if n_candles >= 201 else s200
        sma200_slope = float(s200 - s200_prev) if ok(s200) and ok(s200_prev) else 0
        sma_label = "SMA(200)"
        sma_max_pts = 25
    elif has_sma50:
        s200 = calc_sma(close, 50).iloc[-1]
        s200_prev = calc_sma(close, 50).iloc[-2] if n_candles >= 51 else s200
        sma200_slope = float(s200 - s200_prev) if ok(s200) and ok(s200_prev) else 0
        sma_label = "SMA(50) fallback"
        sma_max_pts = 15
    else:
        s200 = np.nan; s200_prev = np.nan; sma200_slope = 0; sma_label = "N/A"; sma_max_pts = 0

    # EMA(21) for near-term proximity scoring
    ema21_val = np.nan
    ema21_pct_diff = 0  # how far price is from EMA(21), as %
    if n_candles >= 21:
        ema21_series = calc_ema(close, 21)
        ema21_val = ema21_series.iloc[-1]
        if ok(ema21_val) and ema21_val > 0:
            ema21_pct_diff = round((price - ema21_val) / ema21_val * 100, 2)

    r14 = calc_rsi(close,14).iloc[-1]
    bm,bu,bl = calc_bb(close,20,2)
    ml,ms,mh = calc_macd(close)
    ov = calc_obv(close,vol)
    adx_s, plus_di_s, minus_di_s = calc_adx(high,low,close,14)

    cbm,cbu,cbl = bm.iloc[-1],bu.iloc[-1],bl.iloc[-1]
    cm,cms,cmh = ml.iloc[-1],ms.iloc[-1],mh.iloc[-1]

    cm_prev = ml.iloc[-2] if len(ml)>=2 else cm
    cm_prev2 = ml.iloc[-3] if len(ml)>=3 else cm_prev
    cm_prev3 = ml.iloc[-4] if len(ml)>=4 else cm_prev2
    macd_slope = float(cm - cm_prev) if ok(cm) and ok(cm_prev) else 0
    macd_slope_prev = float(cm_prev - cm_prev2) if ok(cm_prev) and ok(cm_prev2) else 0
    macd_slope_prev2 = float(cm_prev2 - cm_prev3) if ok(cm_prev2) and ok(cm_prev3) else 0
    macd_accel = macd_slope - macd_slope_prev

    MIN_SLOPE_MAG = 0.01
    was_negative_2d = bool(macd_slope_prev <= -MIN_SLOPE_MAG and macd_slope_prev2 <= -MIN_SLOPE_MAG)
    was_positive_2d = bool(macd_slope_prev >= MIN_SLOPE_MAG and macd_slope_prev2 >= MIN_SLOPE_MAG)
    was_negative_weak = bool(macd_slope_prev <= 0 and macd_slope_prev2 <= 0 and
                              (abs(macd_slope_prev) + abs(macd_slope_prev2)) > MIN_SLOPE_MAG)
    was_positive_weak = bool(macd_slope_prev >= 0 and macd_slope_prev2 >= 0 and
                              (abs(macd_slope_prev) + abs(macd_slope_prev2)) > MIN_SLOPE_MAG)
    slope_cross_up = bool(macd_slope > 0 and (was_negative_2d or was_negative_weak))
    slope_cross_dn = bool(macd_slope < 0 and (was_positive_2d or was_positive_weak))
    early_buy = bool(macd_slope < 0 and macd_accel > 0)
    early_sell = bool(macd_slope > 0 and macd_accel < 0)

    if slope_cross_up: macd_phase = "BUY FLIP"
    elif early_buy: macd_phase = "EARLY BUY"
    elif slope_cross_dn: macd_phase = "SELL FLIP"
    elif early_sell: macd_phase = "EARLY SELL"
    elif macd_slope > 0 and macd_accel >= 0: macd_phase = "BULLISH"
    elif macd_slope < 0 and macd_accel <= 0: macd_phase = "BEARISH"
    else: macd_phase = "NEUTRAL"

    co = ov.iloc[-1]
    co5 = ov.iloc[-5] if len(ov)>=5 else ov.iloc[0]
    co20 = ov.iloc[-20] if len(ov)>=20 else ov.iloc[0]
    curr_adx = adx_s.iloc[-1] if len(adx_s)>0 and ok(adx_s.iloc[-1]) else 0
    curr_plus_di = float(plus_di_s.iloc[-1]) if len(plus_di_s)>0 and ok(plus_di_s.iloc[-1]) else 0
    curr_minus_di = float(minus_di_s.iloc[-1]) if len(minus_di_s)>0 and ok(minus_di_s.iloc[-1]) else 0
    tlb = any(close.iloc[i]<=bl.iloc[i] for i in range(-5,0) if ok(bl.iloc[i]))
    tub = any(close.iloc[i]>=bu.iloc[i] for i in range(-5,0) if ok(bu.iloc[i]))
    abm = bool(price<=cbm) if ok(cbm) else False

    support, resistance = find_support_resistance(high,low,close, min(60, n_candles - 10))
    risk = round(price-support,2) if support else 0
    reward = round(resistance-price,2) if resistance else 0
    rr_ratio = round(reward/risk,2) if risk>0 else 0

    atr_raw = (high - low).ewm(span=14, adjust=False).mean()
    atr = float(atr_raw.iloc[-1]) if len(atr_raw) > 0 and ok(atr_raw.iloc[-1]) else 0
    risk_in_atrs = round(risk / atr, 2) if atr > 0 and risk > 0 else 0
    RISK_PER_TRADE = 10000
    position_size = int(RISK_PER_TRADE / risk) if risk > 0 else 0
    capital_needed = round(position_size * price, 0) if position_size > 0 else 0
    potential_profit = round(position_size * reward, 0) if position_size > 0 else 0
    roc_pct = round(potential_profit / capital_needed * 100, 2) if capital_needed > 0 else 0

    cm_prev_val = ml.iloc[-2] if len(ml) >= 2 else 0
    macd_zero_cross_up = bool(ok(cm) and cm > 0 and cm < 0.15 and ok(cm_prev_val) and cm_prev_val <= 0)
    macd_zero_cross_dn = bool(ok(cm) and cm < 0 and cm > -0.15 and ok(cm_prev_val) and cm_prev_val >= 0)

    macd_curve = []
    macd_zero_y = 0.5
    n_curve = min(20, len(ml))
    if n_curve > 2:
        raw = [float(ml.iloc[-n_curve + j]) for j in range(n_curve) if ok(ml.iloc[-n_curve + j])]
        if raw:
            mn = min(min(raw), 0)
            mx = max(max(raw), 0)
            rng = mx - mn if mx != mn else 1
            macd_curve = [round((v - mn) / rng, 3) for v in raw]
            macd_zero_y = round((0 - mn) / rng, 3)

    # ── BUY SCORE ──
    buy_score = 0; buy_breakdown = {}

    sma_p = bool(price > s200) if ok(s200) else False
    sma_pts = sma_max_pts if sma_p else 0
    buy_score += sma_pts
    if not has_sma200 and has_sma50:
        sma_desc = f"Close > {sma_label} (newer listing, {n_candles} candles)"
    elif has_sma200:
        sma_desc = "Close > SMA(200)"
    else:
        sma_desc = f"N/A ({n_candles} candles < 50)"
    buy_breakdown["sma200"] = {"pass": sma_p, "pts": sma_pts, "max": sma_max_pts, "val": sf(s200), "desc": sma_desc}

    macd_vals = ml.dropna()
    macd_1y_low = 0; macd_low_pct = 0; macd_pctl = 50
    if len(macd_vals) >= 60:
        stable_vals = macd_vals.iloc[60:]
        recent_vals = stable_vals.iloc[-650:] if len(stable_vals) > 650 else stable_vals
        macd_1y_low = float(recent_vals.min())
        macd_1y_high = float(recent_vals.max())
        macd_range = macd_1y_high - macd_1y_low if macd_1y_high != macd_1y_low else 1
        if macd_1y_low < 0:
            macd_low_pct = round(float(cm) / macd_1y_low * 100, 1)
            macd_low_pct = max(0, min(100, macd_low_pct))
        else:
            macd_low_pct = 0
        macd_pctl = round((float(cm) - macd_1y_low) / macd_range * 100, 1)
    else:
        macd_low_pct = 0; macd_pctl = 50

    golden_buy = bool(macd_low_pct >= 60 and macd_slope <= 0.2 and sma200_slope > 0.1)
    macd_inf_pts = 0; macd_inf_desc = ""
    if golden_buy:
        macd_inf_pts = 30
        macd_inf_desc = f"★ GOLDEN BUY — MACD at {macd_low_pct:.0f}% of 1Y low ({macd_1y_low:.1f}) + flat + SMA↑"
    elif macd_zero_cross_up and macd_accel > 0:
        macd_inf_pts = 30
        macd_inf_desc = "MACD crossed 0↑ + accel ↑ (PRIME ENTRY)"
    elif slope_cross_up:
        macd_inf_pts = 20
        macd_inf_desc = "Slope flipped positive ↑"
    elif early_buy:
        macd_inf_pts = 10
        macd_inf_desc = "Decline slowing (d²>0, d<0)"
    pctl_bonus = 0
    if macd_pctl <= 25 and macd_inf_pts > 0:
        pctl_bonus = 3
        macd_inf_desc += f" +pctl bonus ({macd_pctl:.0f}%ile)"
    macd_inf_pass = macd_inf_pts > 0
    golden_bonus = 10 if golden_buy else 0
    buy_score += macd_inf_pts + golden_bonus + pctl_bonus
    buy_breakdown["macd_inflection"] = {"pass": macd_inf_pass, "pts": macd_inf_pts + golden_bonus + pctl_bonus, "max": 43,
                                         "val": sf(macd_slope, 4), "desc": macd_inf_desc or "No inflection detected"}

    rsi_pts = 0; rsi_desc = ""
    if ok(r14):
        r = float(r14)
        if 25 <= r <= 55:
            if r <= 35: rsi_pts = int(5 + 15 * (r - 25) / 10)
            else: rsi_pts = int(20 * (55 - r) / 20)
            rsi_pts = max(0, min(20, rsi_pts))
            rsi_desc = f"RSI {r:.1f} (graduated, peak at 35)"
        elif r < 25:
            rsi_pts = 3
            rsi_desc = f"RSI {r:.1f} (deeply oversold — caution)"
        else:
            rsi_pts = 0
            rsi_desc = f"RSI {r:.1f} (above buy zone)"
    else:
        rsi_desc = "RSI unavailable"
    rsi_p = rsi_pts > 0
    buy_score += rsi_pts
    buy_breakdown["rsi"] = {"pass": rsi_p, "pts": rsi_pts, "max": 20, "val": sf(r14), "desc": rsi_desc}

    bb_p = bool(abm or tlb) if ok(cbm) else False
    bb_pts = (10 + (5 if tlb else 0)) if bb_p else 0; buy_score += bb_pts
    buy_breakdown["bollinger"] = {"pass": bb_p, "pts": bb_pts, "max": 15, "val": sf(cbm), "desc": "At/below mid BB" + (" +touched lower" if tlb else "")}

    bullish_trend = bool(curr_plus_di > curr_minus_di)
    adx_strong = bool(curr_adx > 25)
    adx_very_strong = bool(curr_adx > 30)
    if adx_strong and bullish_trend:
        adx_pts = 15 + (5 if adx_very_strong else 0)
        adx_desc = f"ADX {float(curr_adx):.1f} · +DI({curr_plus_di:.1f}) > -DI({curr_minus_di:.1f}) ↑ uptrend"
        if adx_very_strong: adx_desc += " +bonus >30"
    elif adx_strong and not bullish_trend:
        adx_pts = 0
        adx_desc = f"ADX {float(curr_adx):.1f} · -DI({curr_minus_di:.1f}) > +DI({curr_plus_di:.1f}) ↓ downtrend — no points"
    else:
        adx_pts = 0
        adx_desc = f"ADX {float(curr_adx):.1f} ≤ 25 — weak trend"
    adx_p = adx_pts > 0
    buy_score += adx_pts
    buy_breakdown["adx"] = {"pass": adx_p, "pts": adx_pts, "max": 20, "val": sf(curr_adx), "desc": adx_desc,
                             "plus_di": sf(curr_plus_di, 1), "minus_di": sf(curr_minus_di, 1)}

    obv_p = bool(co > co5 and co > co20)
    obv_pts = 5 if obv_p else 0; buy_score += obv_pts
    buy_breakdown["obv"] = {"pass": obv_p, "pts": obv_pts, "max": 5, "val": sf(co, 0), "desc": "OBV rising vs 5d & 20d"}

    # 7. EMA(21) Proximity — max 10 pts
    #    Rewards stocks pulled back near their 21-day EMA (ideal entry zone)
    #    Penalizes stocks stretched far above (chasing) or with no EMA data
    ema_pts = 0; ema_desc = ""
    if ok(ema21_val) and n_candles >= 21:
        d = ema21_pct_diff
        if d < -3:
            # More than 3% below EMA(21) — falling hard, risky
            ema_pts = 3
            ema_desc = f"Price {d:.1f}% below EMA(21) — deep pullback, caution"
        elif -3 <= d < 0:
            # 0-3% below EMA(21) — dipped below, pullback buy zone
            ema_pts = 8
            ema_desc = f"Price {d:.1f}% below EMA(21) — pullback entry zone"
        elif 0 <= d <= 2:
            # 0-2% above EMA(21) — sitting right on it, ideal
            ema_pts = 10
            ema_desc = f"Price {d:.1f}% above EMA(21) — ideal proximity"
        elif 2 < d <= 4:
            # 2-4% above — slightly stretched
            ema_pts = 5
            ema_desc = f"Price {d:.1f}% above EMA(21) — slightly stretched"
        elif 4 < d <= 7:
            # 4-7% above — stretched
            ema_pts = 2
            ema_desc = f"Price {d:.1f}% above EMA(21) — stretched"
        else:
            # >7% above — very stretched, wait for pullback
            ema_pts = 0
            ema_desc = f"Price {d:.1f}% above EMA(21) — overextended, wait"
    else:
        ema_desc = "EMA(21) unavailable (< 21 candles)"
    ema_p = ema_pts > 0
    buy_score += ema_pts
    buy_breakdown["ema21"] = {"pass": ema_p, "pts": ema_pts, "max": 10,
                               "val": sf(ema21_val), "desc": ema_desc,
                               "pct_diff": ema21_pct_diff}

    buy_signal = "STRONG BUY" if buy_score >= 75 else ("MODERATE BUY" if buy_score >= 60 else "NO SIGNAL")

    # ── SELL SCORE ──
    sell_score = 0; sell_breakdown = {}

    sma_sell_p = bool(price < s200) if ok(s200) else False
    sma_sell_pts = 25 if sma_sell_p else 0
    sell_score += sma_sell_pts
    sell_breakdown["sma_break"] = {"pass": sma_sell_p, "pts": sma_sell_pts, "max": 25,
        "val": sf(s200), "desc": "Close < SMA(200) — below long-term trend" if sma_sell_p else "Price above SMA(200) — trend intact"}

    sell_macd_pts = 0; sell_macd_desc = ""
    if macd_zero_cross_dn and macd_accel < 0:
        sell_macd_pts = 25; sell_macd_desc = "MACD crossed 0↓ + deceleration (PRIME EXIT)"
    elif slope_cross_dn:
        sell_macd_pts = 18; sell_macd_desc = "Slope flipped negative ↓ (SELL FLIP)"
    elif early_sell:
        sell_macd_pts = 8; sell_macd_desc = "Rise slowing (d²<0, d>0) — early warning"
    sell_pctl_bonus = 0
    if macd_pctl >= 75 and sell_macd_pts > 0:
        sell_pctl_bonus = 3; sell_macd_desc += f" +pctl bonus ({macd_pctl:.0f}%ile)"
    sell_macd_pass = sell_macd_pts > 0
    sell_score += sell_macd_pts + sell_pctl_bonus
    sell_breakdown["macd_inflection"] = {"pass": sell_macd_pass, "pts": sell_macd_pts + sell_pctl_bonus, "max": 28,
        "val": sf(macd_slope, 4), "desc": sell_macd_desc or "No bearish inflection detected"}

    rsi_sell_pts = 0; rsi_sell_desc = ""
    if ok(r14):
        r = float(r14)
        if r > 85: rsi_sell_pts = 10; rsi_sell_desc = f"RSI {r:.1f} — extremely overbought"
        elif r >= 70: rsi_sell_pts = 20; rsi_sell_desc = f"RSI {r:.1f} — overbought (max pts)"
        elif r >= 65: rsi_sell_pts = 15; rsi_sell_desc = f"RSI {r:.1f} — approaching overbought"
        elif r >= 60: rsi_sell_pts = 8; rsi_sell_desc = f"RSI {r:.1f} — elevated, early warning"
        else: rsi_sell_pts = 0; rsi_sell_desc = f"RSI {r:.1f} (below sell zone)"
    sell_score += rsi_sell_pts
    sell_breakdown["rsi"] = {"pass": rsi_sell_pts > 0, "pts": rsi_sell_pts, "max": 20, "val": sf(r14), "desc": rsi_sell_desc}

    bb_sell_upper = bool(price >= cbu) if ok(cbu) else False
    bb_sell_pts = 10 if bb_sell_upper else (5 if tub else 0)
    sell_score += bb_sell_pts
    sell_breakdown["bollinger"] = {"pass": bb_sell_pts > 0, "pts": bb_sell_pts, "max": 10,
        "val": sf(cbu), "desc": ("At/above upper BB" if bb_sell_upper else "Touched upper BB recently") if bb_sell_pts > 0 else "Below upper band"}

    bearish_trend = bool(curr_minus_di > curr_plus_di)
    adx_sell_pts = 0; adx_sell_desc = ""
    if curr_adx > 25 and bearish_trend:
        adx_sell_pts = 10 + (5 if curr_adx > 30 else 0)
        adx_sell_desc = f"ADX {float(curr_adx):.1f} · -DI > +DI ↓ confirmed downtrend"
    elif curr_adx > 25 and not bearish_trend:
        adx_sell_desc = f"ADX {float(curr_adx):.1f} · +DI > -DI — uptrend, no sell pts"
    else:
        adx_sell_desc = f"ADX {float(curr_adx):.1f} ≤ 25 — weak trend"
    sell_score += adx_sell_pts
    sell_breakdown["adx"] = {"pass": adx_sell_pts > 0, "pts": adx_sell_pts, "max": 15, "val": sf(curr_adx), "desc": adx_sell_desc}

    obv_sell_p = bool(co < co5 and co < co20)
    obv_sell_pts = 10 if obv_sell_p else 0
    sell_score += obv_sell_pts
    sell_breakdown["obv"] = {"pass": obv_sell_p, "pts": obv_sell_pts, "max": 10,
        "val": sf(co, 0), "desc": "OBV falling vs 5d & 20d — distribution" if obv_sell_p else "OBV holding or rising"}

    has_structural_sell = sma_sell_p or (adx_sell_pts > 0)
    if sell_score >= 65 and has_structural_sell: sell_signal = "STRONG SELL"
    elif sell_score >= 45: sell_signal = "MODERATE SELL"
    else: sell_signal = "NO SIGNAL"

    p0 = float(close.iloc[-1])
    p1 = float(close.iloc[-2]) if n_candles >= 2 else p0
    p2 = float(close.iloc[-3]) if n_candles >= 3 else p1
    p3 = float(close.iloc[-4]) if n_candles >= 4 else p2
    roc3 = ((p0 - p3) / p3 * 100) if p3 > 0 else 0
    price_vel = ((p0 - p1) / p1 * 100) if p1 > 0 else 0
    price_vel_prev = ((p1 - p2) / p2 * 100) if p2 > 0 else 0
    price_accel = price_vel - price_vel_prev
    up_days = 0
    for idx in range(1, min(6, n_candles)):
        if float(close.iloc[-idx]) > float(close.iloc[-idx-1]):
            up_days += 1
        else:
            break

    return {
        "price":sf(price),"sma200":sf(s200),"sma200_slope":round(sma200_slope,3),
        "sma_label":sma_label,"rsi":sf(r14),
        "bb_upper":sf(cbu),"bb_mid":sf(cbm),"bb_lower":sf(cbl),
        "macd":sf(cm,4),"macd_signal":sf(cms,4),"macd_hist":sf(cmh,4),
        "macd_slope":round(macd_slope,4),"macd_accel":round(macd_accel,4),"macd_pctl":macd_pctl,"macd_low_pct":macd_low_pct,"macd_1y_low":round(macd_1y_low,2),
        "macd_phase":macd_phase,"macd_curve":macd_curve,"macd_zero_y":macd_zero_y,
        "obv":sf(co,0),"adx":sf(curr_adx),"plus_di":sf(curr_plus_di,1),"minus_di":sf(curr_minus_di,1),
        "adx_bullish":bullish_trend,
        "ema21":sf(ema21_val),"ema21_pct_diff":ema21_pct_diff,
        "avg_vol_20":round(avg_vol_20, 0),
        "price_roc3":round(roc3,2),"price_vel":round(price_vel,2),
        "price_accel":round(price_accel,3),"up_days":up_days,
        "support":support,"resistance":resistance,"risk":risk,"reward":reward,"rr_ratio":rr_ratio,
        "atr":round(atr,2),"risk_in_atrs":risk_in_atrs,"position_size":position_size,
        "capital_needed":capital_needed,"potential_profit":potential_profit,"roc_pct":roc_pct,
        "currency": currency_symbol,
        "buy_score":buy_score,"buy_signal":buy_signal,"buy_breakdown":buy_breakdown,
        "sell_score":sell_score,"sell_signal":sell_signal,"sell_breakdown":sell_breakdown,
        "is_buy":buy_score>=75,"is_moderate_buy":buy_score>=60,
        "is_sell":sell_score>=65,"is_moderate_sell":sell_score>=45,
        "golden_buy":golden_buy,
    }

# ═══════════════════════════════════════════════════════════════
# SCAN — NIFTY 500 (Angel One)
# ═══════════════════════════════════════════════════════════════

def run_nifty_scan():
    global scan_results, scan_progress, current_pace
    if not credentials_ok:
        logger.error("Cannot scan NIFTY — not logged in")
        return
    if is_scanning["nifty500"]:
        logger.warning("NIFTY scan already running")
        return
    is_scanning["nifty500"] = True
    abort_scan["nifty500"] = False
    current_pace = PACE_MIN
    mkt = "nifty500"
    scan_results[mkt]["status"] = "scanning"

    total = len(scan_instrument_list)
    scan_progress[mkt] = {"current": 0, "total": total, "ok": 0, "errors": 0, "skipped": 0, "rate_limited": False, "pace": current_pace}
    logger.info(f"Starting NIFTY scan: {total} stocks")
    t0 = time.time()
    buys, sells, all_s, errs = [], [], [], []

    portfolio = fetch_portfolio()
    ptokens = {h["token"] for h in portfolio}
    consecutive_errors = 0

    for i, inst in enumerate(scan_instrument_list):
        if abort_scan["nifty500"]: break
        sym, token = inst["symbol"], inst["token"]
        name = inst.get("name", sym.replace("-EQ",""))
        clean = sym.replace("-EQ","")

        scan_progress[mkt]["current"] = i + 1
        scan_progress[mkt]["pace"] = round(current_pace, 2)

        if consecutive_errors >= 50:
            scan_progress[mkt]["rate_limited"] = True
            break

        try:
            result = fetch_candle_data(token)
            if result is RATE_LIMITED:
                consecutive_errors += 1
                errs.append({"symbol": sym, "error": "Rate limited"})
                scan_progress[mkt]["errors"] = len(errs)
                current_pace = min(current_pace * PACE_BACKOFF_MULT, PACE_MAX)
                time.sleep(5)
                try: create_session()
                except: pass
                continue

            if result is None:
                scan_progress[mkt]["skipped"] = scan_progress[mkt].get("skipped", 0) + 1
                consecutive_errors = 0
                current_pace = max(current_pace * PACE_RECOVERY_MULT, PACE_MIN)
                continue

            df = result
            if len(df) < 50:
                scan_progress[mkt]["skipped"] = scan_progress[mkt].get("skipped", 0) + 1
                consecutive_errors = 0; current_pace = max(current_pace * PACE_RECOVERY_MULT, PACE_MIN)
                continue

            a = analyze_stock(df, currency_symbol="₹")
            if a is None:
                scan_progress[mkt]["skipped"] = scan_progress[mkt].get("skipped", 0) + 1
                consecutive_errors = 0; current_pace = max(current_pace * PACE_RECOVERY_MULT, PACE_MIN)
                continue

            consecutive_errors = 0
            current_pace = max(current_pace * PACE_RECOVERY_MULT, PACE_MIN)

            indices = get_tags_for(clean, INDEX_TAGS)
            stock = {"symbol": sym, "name": name, "token": token,
                     "in_portfolio": token in ptokens, "indices": indices, "market": "nifty500", **a}
            all_s.append(stock)
            scan_progress[mkt]["ok"] = len(all_s)

            if a["is_buy"] or a["is_moderate_buy"]: buys.append(stock)
            if (a["is_sell"] or a["is_moderate_sell"]) and token in ptokens: sells.append(stock)

            if len(all_s) % 10 == 0:
                scan_results[mkt].update({
                    "total_scanned": len(all_s),
                    "buy_signals": sorted(buys, key=lambda x: x["buy_score"], reverse=True),
                    "sell_signals": sorted(sells, key=lambda x: x["sell_score"], reverse=True),
                    "all_stocks": sorted(all_s, key=lambda x: x["buy_score"], reverse=True),
                })

            time.sleep(current_pace)
        except Exception as e:
            err_str = str(e)
            errs.append({"symbol": sym, "error": err_str})
            scan_progress[mkt]["errors"] = len(errs)
            if "rate" in err_str.lower() or "timeout" in err_str.lower():
                consecutive_errors += 1
                current_pace = min(current_pace * PACE_BACKOFF_MULT, PACE_MAX)
                time.sleep(3)
            else:
                consecutive_errors = 0

        if (i+1) % 150 == 0:
            try: create_session(); time.sleep(1)
            except: pass

    elapsed = time.time() - t0
    scan_results[mkt] = {
        "last_scan": datetime.now().isoformat(), "status": "complete",
        "scan_duration_sec": round(elapsed,1), "total_scanned": len(all_s),
        "buy_signals": sorted(buys, key=lambda x: x["buy_score"], reverse=True),
        "sell_signals": sorted(sells, key=lambda x: x["sell_score"], reverse=True),
        "all_stocks": sorted(all_s, key=lambda x: x["buy_score"], reverse=True),
        "portfolio_holdings": portfolio, "errors": errs,
    }
    logger.info(f"NIFTY SCAN DONE in {elapsed/60:.1f}min — {len(all_s)} stocks, {len(buys)} buys")
    is_scanning["nifty500"] = False

# ═══════════════════════════════════════════════════════════════
# SCAN — NASDAQ 100 (yfinance)
# ═══════════════════════════════════════════════════════════════

def run_nasdaq_scan():
    global scan_results
    mkt = "nasdaq100"
    if is_scanning["nasdaq100"]:
        logger.warning("NASDAQ scan already running")
        return
    is_scanning["nasdaq100"] = True
    abort_scan["nasdaq100"] = False
    scan_results[mkt]["status"] = "scanning"

    symbols = get_nasdaq100_symbols()
    total = len(symbols)
    scan_progress[mkt] = {"current": 0, "total": total, "ok": 0, "errors": 0, "skipped": 0, "rate_limited": False, "pace": 0.5}
    logger.info(f"Starting NASDAQ scan: {total} stocks")
    t0 = time.time()
    buys, sells, all_s, errs = [], [], [], []

    for i, sym in enumerate(symbols):
        if abort_scan["nasdaq100"]: break
        scan_progress[mkt]["current"] = i + 1

        try:
            df = fetch_us_candle_data(sym)
            if df is US_RATE_LIMITED:
                errs.append({"symbol": sym, "error": "Rate limited"})
                scan_progress[mkt]["errors"] = len(errs)
                time.sleep(5)
                continue

            if df is None or len(df) < 50:
                scan_progress[mkt]["skipped"] = scan_progress[mkt].get("skipped", 0) + 1
                continue

            a = analyze_stock(df, currency_symbol="$", min_avg_volume=50000)
            if a is None:
                scan_progress[mkt]["skipped"] = scan_progress[mkt].get("skipped", 0) + 1
                continue

            stock = {"symbol": sym, "name": sym, "token": sym,
                     "in_portfolio": False, "indices": ["NASDAQ 100"], "market": "nasdaq100", **a}
            all_s.append(stock)
            scan_progress[mkt]["ok"] = len(all_s)

            if a["is_buy"] or a["is_moderate_buy"]: buys.append(stock)
            if a["is_sell"] or a["is_moderate_sell"]: sells.append(stock)

            if len(all_s) % 10 == 0:
                scan_results[mkt].update({
                    "total_scanned": len(all_s),
                    "buy_signals": sorted(buys, key=lambda x: x["buy_score"], reverse=True),
                    "sell_signals": sorted(sells, key=lambda x: x["sell_score"], reverse=True),
                    "all_stocks": sorted(all_s, key=lambda x: x["buy_score"], reverse=True),
                })

            time.sleep(0.4)  # gentle pacing for Yahoo
        except Exception as e:
            errs.append({"symbol": sym, "error": str(e)})
            scan_progress[mkt]["errors"] = len(errs)

    elapsed = time.time() - t0
    scan_results[mkt] = {
        "last_scan": datetime.now().isoformat(), "status": "complete",
        "scan_duration_sec": round(elapsed,1), "total_scanned": len(all_s),
        "buy_signals": sorted(buys, key=lambda x: x["buy_score"], reverse=True),
        "sell_signals": sorted(sells, key=lambda x: x["sell_score"], reverse=True),
        "all_stocks": sorted(all_s, key=lambda x: x["buy_score"], reverse=True),
        "portfolio_holdings": [], "errors": errs,
    }
    logger.info(f"NASDAQ SCAN DONE in {elapsed/60:.1f}min — {len(all_s)} stocks, {len(buys)} buys")
    is_scanning["nasdaq100"] = False

# ═══════════════════════════════════════════════════════════════
# COMBINED SCAN (parallel)
# ═══════════════════════════════════════════════════════════════

def run_full_scan(markets=None):
    """Launch scans for requested markets. Each market runs independently."""
    if markets is None:
        markets = ["nifty500", "nasdaq100"]

    threads = []
    if "nifty500" in markets and credentials_ok and not is_scanning["nifty500"]:
        threads.append(threading.Thread(target=run_nifty_scan, daemon=True))
    if "nasdaq100" in markets and not is_scanning["nasdaq100"]:
        threads.append(threading.Thread(target=run_nasdaq_scan, daemon=True))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    logger.info("Requested market scans complete.")

# ═══════════════════════════════════════════════════════════════
# SCHEDULER
# ═══════════════════════════════════════════════════════════════

def scheduler():
    while True:
        now = datetime.now()
        wd = now.weekday() < 5
        # Indian market hours
        ao = (now.hour==9 and now.minute>=15) or now.hour>=10
        bc = now.hour<15 or (now.hour==15 and now.minute<=30)
        if wd and ao and bc:
            run_full_scan(["nifty500"])
        # US market scan at 8 PM IST (US market open ~7 PM IST)
        if wd and now.hour == 20 and now.minute < 20:
            run_full_scan(["nasdaq100"])
        time.sleep(SCAN_INTERVAL)

# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route("/api/auth", methods=["POST"])
def api_auth():
    body = request.get_json() or {}
    pwd = body.get("password", "").strip()
    if pwd == APP_PASSWORD:
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "Incorrect password"}), 401

@app.route("/api/docs")
def api_docs():
    doc_path = os.path.join(os.path.dirname(__file__), "INDICATORS.md")
    if os.path.exists(doc_path):
        with open(doc_path, "r", encoding="utf-8") as f:
            return jsonify({"content": f.read()})
    return jsonify({"content": "# Documentation\n\nINDICATORS.md not found."})

@app.route("/api/status")
def api_status():
    return jsonify({
        "connected": credentials_ok,
        "instruments_scan": len(scan_instrument_list),
        "last_scan": scan_results.get("nifty500", {}).get("last_scan"),
        "scan_status": "scanning" if any(is_scanning.values()) else "idle",
        "is_scanning": any(is_scanning.values()),
        "is_scanning_per_market": is_scanning,
        "missing": check_credentials(),
        "zerodha_connected": zerodha_access_token is not None,
        "zerodha_user": zerodha_user.get("user_name","") if zerodha_user else "",
        "zerodha_configured": bool(ZERODHA_API_KEY and ZERODHA_API_SECRET),
        "login_url": f"https://kite.zerodha.com/connect/login?v=3&api_key={ZERODHA_API_KEY}" if ZERODHA_API_KEY else "",
        "markets": {
            "nifty500": {"available": credentials_ok, "stocks": len(scan_instrument_list),
                         "last_scan": scan_results["nifty500"].get("last_scan"),
                         "status": scan_results["nifty500"].get("status", "not_started")},
            "nasdaq100": {"available": True, "stocks": len(get_nasdaq100_symbols()),
                          "last_scan": scan_results["nasdaq100"].get("last_scan"),
                          "status": scan_results["nasdaq100"].get("status", "not_started")},
        }
    })

@app.route("/api/reconnect", methods=["POST"])
def api_reconnect():
    ok = create_session()
    if ok:
        if len(instrument_list) == 0:
            fetch_instrument_list()
        return jsonify({"status": "connected", "instruments_scan": len(scan_instrument_list)})
    return jsonify({"error": "Login failed. Check credentials or try again in a minute."}), 500

@app.route("/api/stop", methods=["POST"])
def api_stop_scan():
    """Abort all running scans, or a specific market via ?market=nifty500"""
    market = request.args.get("market", None)
    if market and market in abort_scan:
        abort_scan[market] = True
        return jsonify({"status": "stopping", "market": market})
    # Stop all
    if not any(is_scanning.values()):
        return jsonify({"status": "not_scanning"})
    for m in abort_scan:
        abort_scan[m] = True
    return jsonify({"status": "stopping_all"})

@app.route("/api/reset", methods=["POST"])
def api_reset():
    global scan_results, scan_progress
    # Stop all scans
    for m in abort_scan:
        abort_scan[m] = True
    for _ in range(20):
        if not any(is_scanning.values()): break
        time.sleep(0.25)
    for mkt in ["nifty500", "nasdaq100"]:
        scan_results[mkt] = {
            "last_scan": None, "status": "not_started", "total_scanned": 0,
            "buy_signals": [], "sell_signals": [], "all_stocks": [],
            "portfolio_holdings": [], "errors": [],
        }
        scan_progress[mkt] = {"current": 0, "total": 0, "ok": 0, "errors": 0, "skipped": 0, "rate_limited": False}
    for m in abort_scan:
        abort_scan[m] = False
    create_session()
    if len(instrument_list) == 0:
        fetch_instrument_list()
    return jsonify({"status": "reset", "connected": credentials_ok})

@app.route("/api/scan", methods=["POST"])
def api_trigger_scan():
    body = request.get_json() or {}
    markets = body.get("markets", ["nifty500", "nasdaq100"])
    if isinstance(markets, str):
        markets = [markets]

    if "nifty500" in markets:
        if not credentials_ok:
            if not create_session():
                markets = [m for m in markets if m != "nifty500"]
        if len(scan_instrument_list) == 0:
            fetch_instrument_list()

    # Filter out markets already scanning
    already = [m for m in markets if is_scanning.get(m, False)]
    remaining = [m for m in markets if not is_scanning.get(m, False)]
    if not remaining:
        return jsonify({"status": "already_scanning", "markets": already}), 409
    markets = remaining

    threading.Thread(target=run_full_scan, args=(markets,), daemon=True).start()
    return jsonify({"status": "scan_started", "markets": markets})

@app.route("/api/results")
def api_results():
    market = request.args.get("market", "nifty500")
    if market in scan_results:
        return jsonify(scan_results[market])
    return jsonify({"error": "Unknown market"}), 400

@app.route("/api/progress")
def api_progress():
    return jsonify(scan_progress)

@app.route("/api/stock/<symbol>")
def api_stock_detail(symbol):
    for mkt in scan_results:
        for s in scan_results[mkt].get("all_stocks", []):
            if s["symbol"] == symbol or s["name"] == symbol:
                return jsonify(s)
    return jsonify({"error": "Not found"}), 404

@app.route("/api/index_chart")
def api_index_chart():
    """Fetch index chart data. Params: index (^NSEI, ^IXIC, ^NSEBANK), resolution (1d, 1wk, 1mo)"""
    index_symbol = request.args.get("index", "^NSEI")
    resolution = request.args.get("resolution", "1d")
    period = request.args.get("period", "1y")
    # Retry up to 2 times — Render's outbound connections can be flaky on first attempt
    for attempt in range(2):
        try:
            data = fetch_index_chart_data(index_symbol, resolution, period)
            if data is not None:
                return jsonify(data)
            if attempt == 0:
                logger.warning(f"Index chart attempt 1 failed for {index_symbol}, retrying...")
                time.sleep(1)
        except Exception as e:
            logger.error(f"Index chart error (attempt {attempt+1}): {e}")
            if attempt == 0:
                time.sleep(1)
    return jsonify({"error": "Could not fetch index data. Yahoo Finance may be temporarily unavailable."}), 500

SECTOR_INDICES = [
    {"symbol": "^CNXBANK", "name": "NIFTY Bank", "sector": "Banking"},
    {"symbol": "^CNXIT", "name": "NIFTY IT", "sector": "IT / Tech"},
    {"symbol": "^CNXPHARMA", "name": "NIFTY Pharma", "sector": "Pharma"},
    {"symbol": "^CNXFMCG", "name": "NIFTY FMCG", "sector": "FMCG"},
    {"symbol": "^CNXAUTO", "name": "NIFTY Auto", "sector": "Auto"},
    {"symbol": "^CNXENERGY", "name": "NIFTY Energy", "sector": "Energy"},
    {"symbol": "^CNXFIN", "name": "NIFTY Fin Service", "sector": "Fin Services"},
    {"symbol": "^CNXMETAL", "name": "NIFTY Metal", "sector": "Metals"},
    {"symbol": "^CNXREALTY", "name": "NIFTY Realty", "sector": "Realty"},
    {"symbol": "^CNXMEDIA", "name": "NIFTY Media", "sector": "Media"},
]

@app.route("/api/sectors")
def api_sectors():
    """Fetch all sector indices in one call. Returns list of chart data objects."""
    resolution = request.args.get("resolution", "1d")
    period = request.args.get("period", "3mo")
    results = []
    for sec in SECTOR_INDICES:
        try:
            data = fetch_index_chart_data(sec["symbol"], resolution, period)
            if data:
                data["sector"] = sec["sector"]
                results.append(data)
            else:
                results.append({"symbol": sec["symbol"], "name": sec["name"],
                                "sector": sec["sector"], "error": True})
        except Exception as e:
            results.append({"symbol": sec["symbol"], "name": sec["name"],
                            "sector": sec["sector"], "error": True})
    return jsonify(results)

@app.route("/")
def serve_frontend():
    return send_from_directory(".", "index.html")

# ═══════════════════════════════════════════════════════════════
# ZERODHA (unchanged from v3.1)
# ═══════════════════════════════════════════════════════════════

def _zerodha_headers():
    return {"Authorization": f"token {ZERODHA_API_KEY}:{zerodha_access_token}"}

def _verify_zerodha_token():
    global zerodha_user
    if not zerodha_access_token or not ZERODHA_API_KEY: return None
    try:
        resp = requests.get("https://api.kite.trade/user/profile", headers=_zerodha_headers(), timeout=10)
        data = resp.json()
        if data.get("status") == "success":
            zerodha_user = data["data"]
            return zerodha_user
        return None
    except: return None

@app.route("/api/zerodha/status")
def zerodha_status():
    return jsonify({
        "configured": bool(ZERODHA_API_KEY and ZERODHA_API_SECRET),
        "connected": zerodha_access_token is not None,
        "user_name": zerodha_user.get("user_name", "") if zerodha_user else "",
        "user_id": zerodha_user.get("user_id", "") if zerodha_user else "",
        "login_url": f"https://kite.zerodha.com/connect/login?v=3&api_key={ZERODHA_API_KEY}" if ZERODHA_API_KEY else "",
    })

@app.route("/api/zerodha/exchange_token", methods=["POST"])
def zerodha_exchange_token():
    global zerodha_access_token, zerodha_user
    body = request.get_json() or {}
    request_token = body.get("request_token", "").strip()
    if not request_token: return jsonify({"error": "No request_token"}), 400
    if not ZERODHA_API_KEY or not ZERODHA_API_SECRET: return jsonify({"error": "Not configured"}), 400
    try:
        checksum = hashlib.sha256((ZERODHA_API_KEY + request_token + ZERODHA_API_SECRET).encode("utf-8")).hexdigest()
        resp = requests.post("https://api.kite.trade/session/token", data={"api_key": ZERODHA_API_KEY, "request_token": request_token, "checksum": checksum}, timeout=15)
        data = resp.json()
        if data.get("status") != "success": return jsonify({"error": data.get("message", "Unknown error")}), 400
        zerodha_access_token = data["data"]["access_token"]
        zerodha_user = data["data"]
        return jsonify({"status": "connected", "user_name": data["data"].get("user_name", "")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/zerodha/set_token", methods=["POST"])
def zerodha_set_token():
    global zerodha_access_token, zerodha_user
    body = request.get_json() or {}
    token = body.get("access_token", "").strip()
    if not token: return jsonify({"error": "No token"}), 400
    zerodha_access_token = token
    user = _verify_zerodha_token()
    if user: return jsonify({"status": "connected", "user_name": user.get("user_name", "")})
    zerodha_access_token = None
    return jsonify({"error": "Token invalid"}), 401

@app.route("/api/zerodha/logout", methods=["POST"])
def zerodha_logout():
    global zerodha_access_token, zerodha_user
    zerodha_access_token = None; zerodha_user = None
    return jsonify({"status": "logged_out"})

@app.route("/api/zerodha/holdings")
def zerodha_holdings():
    if not zerodha_access_token: return jsonify({"error": "Not connected"}), 401
    try:
        resp = requests.get("https://api.kite.trade/portfolio/holdings", headers=_zerodha_headers(), timeout=15)
        data = resp.json()
        if data.get("status") != "success":
            err_msg = data.get("message", "Unknown error")
            if "token" in err_msg.lower(): return jsonify({"error": "Session expired", "expired": True}), 401
            return jsonify({"error": err_msg}), 400
        raw_holdings = data.get("data", [])
        holdings = []
        scan_lookup = {}
        for s in scan_results["nifty500"].get("all_stocks", []):
            scan_lookup[s["symbol"].replace("-EQ", "")] = s
        for h in raw_holdings:
            tsym = h.get("tradingsymbol", "")
            qty = h.get("quantity", 0)
            if qty <= 0: continue
            avg_price = float(h.get("average_price", 0))
            ltp = float(h.get("last_price", 0))
            invested = avg_price * qty; current_val = ltp * qty
            total_return_pct = ((current_val - invested) / invested * 100) if invested > 0 else 0
            holding = {
                "tradingsymbol": tsym, "exchange": h.get("exchange", "NSE"),
                "quantity": qty, "avg_price": round(avg_price, 2), "ltp": round(ltp, 2),
                "day_change": round(float(h.get("day_change", 0)), 2),
                "day_change_pct": round(float(h.get("day_change_percentage", 0)), 2),
                "pnl": round(float(h.get("pnl", 0)), 2),
                "invested": round(invested, 2), "current_val": round(current_val, 2),
                "total_return_pct": round(total_return_pct, 2),
                "scan": scan_lookup.get(tsym),
            }
            holdings.append(holding)
        holdings.sort(key=lambda x: x["current_val"], reverse=True)
        total_invested = sum(h["invested"] for h in holdings)
        total_current = sum(h["current_val"] for h in holdings)
        total_pnl = total_current - total_invested
        day_pnl = sum(h["day_change"] * h["quantity"] for h in holdings)
        sell_alerts = [h for h in holdings if h["scan"] and (h["scan"].get("is_sell") or h["scan"].get("is_moderate_sell"))]
        return jsonify({
            "holdings": holdings,
            "summary": {
                "total_stocks": len(holdings), "total_invested": round(total_invested, 2),
                "total_current": round(total_current, 2), "total_pnl": round(total_pnl, 2),
                "total_return_pct": round((total_pnl / total_invested * 100) if total_invested > 0 else 0, 2),
                "day_pnl": round(day_pnl, 2), "sell_alerts": len(sell_alerts),
                "with_scan_data": sum(1 for h in holdings if h["scan"]),
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ═══════════════════════════════════════════════════════════════
# INIT
# ═══════════════════════════════════════════════════════════════

def initialize():
    global credentials_ok
    missing = check_credentials()
    if missing:
        logger.error(f"Missing env vars: {', '.join(missing)}")
        credentials_ok = False
    else:
        if create_session():
            # Run instrument download in background — poll for completion
            # instead of Thread.join() which has a race condition on Windows/Python 3.12
            logger.info("Loading instruments (max 90s, server starts regardless)...")
            threading.Thread(target=fetch_instrument_list, daemon=True).start()

            # Poll every 0.5s for up to 90s — check if instruments loaded
            for _ in range(180):
                if len(scan_instrument_list) > 0:
                    break
                time.sleep(0.5)

            if len(scan_instrument_list) > 0:
                logger.info(f"Instruments loaded: {len(scan_instrument_list)} stocks")
            else:
                logger.warning("Instrument download still running or failed — server starting anyway.")
                logger.warning("NIFTY scan will work once instruments finish loading. NASDAQ works now.")
        else:
            logger.warning("Initial login failed — server starting anyway.")

    if zerodha_access_token and ZERODHA_API_KEY:
        if _verify_zerodha_token():
            logger.info("Zerodha token valid.")

    threading.Thread(target=scheduler, daemon=True).start()
    logger.info(f"Ready. NIFTY: {len(scan_instrument_list)} stocks, NASDAQ: {len(get_nasdaq100_symbols())} stocks")
    return True