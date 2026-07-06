"""
SignalScope v4.1 — Multi-Market Edition
Scans NIFTY 500 (Yahoo Finance) + NASDAQ 100 (Yahoo Finance)
Daily candles, 6 indicators, weighted scoring, index charts.
No broker API credentials required for market data.

v4.1 Changes:
  - Replaced Angel One API with Yahoo Finance for NIFTY 500 data
  - Indian stocks fetched via yfinance using SYMBOL.NS tickers
  - Portfolio data sourced from Zerodha only
  - No API keys or credentials needed for scanning
"""

import os, sys, time, threading, logging, hashlib, json, math
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd, requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import scoring
from index_data import get_scan_universe, build_index_tags, get_tags_for
from us_market import (
    get_nasdaq100_symbols, fetch_us_candle_data, fetch_index_chart_data,
    US_RATE_LIMITED
)

# Force UTF-8 console output so an un-encodable log line (₹, emoji, box-art) on a
# legacy Windows codepage can't crash the server with UnicodeEncodeError.
for _stream in ("stdout", "stderr"):
    try:
        getattr(sys, _stream).reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Surface exceptions that escape a background thread (scan/prefetch/scheduler) —
# by default Python only prints these to stderr with no context.
def _thread_excepthook(args):
    logger.error("UNCAUGHT exception in thread %r", getattr(args.thread, "name", "?"),
                 exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
try:
    threading.excepthook = _thread_excepthook
except Exception:
    pass

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "900"))

MIN_AVG_VOLUME = int(os.getenv("MIN_AVG_VOLUME", "100000"))

APP_PASSWORD = os.getenv("APP_PASSWORD", "signal2026")

# ── AI features (Value/AI scores are on-demand, not part of bulk scans) ──
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
AI_MODEL = os.getenv("AI_MODEL", "claude-sonnet-5").strip()
# Provider: "openai" (ChatGPT) or "anthropic" (Claude). Auto-picks from whichever
# key is present if unset; defaults to OpenAI otherwise.
AI_PROVIDER = os.getenv("AI_PROVIDER", "").strip().lower()
if not AI_PROVIDER:
    AI_PROVIDER = "openai" if OPENAI_API_KEY else ("anthropic" if ANTHROPIC_API_KEY else "openai")

def ai_is_enabled():
    return (AI_PROVIDER == "openai" and bool(OPENAI_API_KEY)) or \
           (AI_PROVIDER == "anthropic" and bool(ANTHROPIC_API_KEY))

def ai_active_model():
    return OPENAI_MODEL if AI_PROVIDER == "openai" else AI_MODEL

# Zerodha
ZERODHA_API_KEY    = os.getenv("ZERODHA_API_KEY", "")
ZERODHA_API_SECRET = os.getenv("ZERODHA_API_SECRET", "")
_zt = os.getenv("ZERODHA_ACCESS_TOKEN", "").strip()
zerodha_access_token = _zt if (_zt and not _zt.startswith("http") and len(_zt) > 20) else None
zerodha_user = None

INDEX_TAGS = build_index_tags()
SCAN_UNIVERSE = set()

scan_instrument_list = []
is_scanning = {"nifty500": False, "nasdaq100": False}
abort_scan = {"nifty500": False, "nasdaq100": False}

current_pace = 0.5
PACE_MIN = 0.5
PACE_MAX = 8.0
PACE_BACKOFF_MULT = 2.0
PACE_RECOVERY_MULT = 0.95
MAX_RETRIES_PER_STOCK = 2

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
# INSTRUMENTS — built from NIFTY 500 universe (no broker API needed)
# ═══════════════════════════════════════════════════════════════

def load_instrument_list():
    """Build NIFTY 500 scan list from index_data. Each stock mapped to a Yahoo Finance ticker."""
    global scan_instrument_list, SCAN_UNIVERSE
    try:
        SCAN_UNIVERSE = get_scan_universe()
        scan_instrument_list = [
            {"symbol": sym + "-EQ", "name": sym, "yf_symbol": sym + ".NS"}
            for sym in sorted(SCAN_UNIVERSE)
        ]
        logger.info(f"Scan list: {len(scan_instrument_list)} stocks (NIFTY 500 universe via Yahoo Finance)")
        return True
    except Exception as e:
        logger.error(f"Failed to load instrument list: {e}")
        return False

# ═══════════════════════════════════════════════════════════════
# DATA FETCH (Yahoo Finance — no credentials required)
# ═══════════════════════════════════════════════════════════════

RATE_LIMITED = "RATE_LIMITED"

def fetch_candle_data(yf_symbol):
    """
    Fetch 2 years of daily OHLCV data from Yahoo Finance v8 chart API.
    Uses direct HTTP (same approach as fetch_index_chart_data) for reliable 429 detection.
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_symbol}"
    params = {"range": "2y", "interval": "1d", "includePrePost": "false", "events": ""}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0",
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code == 429:
            logger.warning(f"Rate limited (429) on {yf_symbol}")
            return RATE_LIMITED
        if resp.status_code != 200:
            logger.debug(f"HTTP {resp.status_code} for {yf_symbol}")
            return None

        data = resp.json()
        chart = data.get("chart", {}).get("result", [])
        if not chart:
            return None

        result = chart[0]
        timestamps = result.get("timestamp", [])
        quote = result.get("indicators", {}).get("quote", [{}])[0]
        opens_raw  = quote.get("open", [])
        highs_raw  = quote.get("high", [])
        lows_raw   = quote.get("low", [])
        closes_raw = quote.get("close", [])
        vols_raw   = quote.get("volume", [])

        if len(timestamps) < 10:
            return None

        rows = []
        for i, ts in enumerate(timestamps):
            c = closes_raw[i] if i < len(closes_raw) else None
            if c is None:
                continue
            rows.append({
                "DateTime": pd.Timestamp.utcfromtimestamp(ts).tz_localize(None),
                "Open":   float(opens_raw[i]  or c),
                "High":   float(highs_raw[i]  or c),
                "Low":    float(lows_raw[i]   or c),
                "Close":  float(c),
                "Volume": int(vols_raw[i] or 0) if i < len(vols_raw) else 0,
            })

        if len(rows) < 50:
            return None

        df = pd.DataFrame(rows).set_index("DateTime")
        return df

    except Exception as e:
        err_str = str(e).lower()
        if "rate" in err_str or "too many" in err_str or "429" in err_str:
            logger.warning(f"Rate limited on {yf_symbol}: {e}")
            return RATE_LIMITED
        logger.debug(f"Candle error {yf_symbol}: {e}")
        return None

def fetch_portfolio():
    """Fetch holdings from Zerodha. Returns list of dicts with tradingsymbol."""
    if not zerodha_access_token or not ZERODHA_API_KEY:
        return []
    try:
        resp = requests.get(
            "https://api.kite.trade/portfolio/holdings",
            headers={"Authorization": f"token {ZERODHA_API_KEY}:{zerodha_access_token}"},
            timeout=15,
        )
        data = resp.json()
        if data.get("status") != "success":
            return []
        return [
            {"symbol": h.get("tradingsymbol", ""), "quantity": h.get("quantity", 0),
             "avg_price": float(h.get("average_price", 0)), "ltp": float(h.get("last_price", 0)),
             "pnl": float(h.get("pnl", 0))}
            for h in data.get("data", []) if h.get("quantity", 0) > 0
        ]
    except Exception as e:
        logger.error(f"Portfolio fetch error: {e}")
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

    # ── MACD percentile / golden-buy inputs (feed both sell score and new scorers) ──
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

    # ═══════════════════════════════════════════════════════════════
    # NEW: MOMENTUM + PULLBACK SCORES (ported from Android — see scoring.py)
    # Trend-continuation and buy-the-dip are scored on SEPARATE lenses so
    # they no longer interfere. Everything below reuses indicators already
    # computed above, plus a few extra long-term/EMA-cascade metrics.
    # ═══════════════════════════════════════════════════════════════
    bullish_trend = bool(curr_plus_di > curr_minus_di)

    # --- EMA(50/200) + slopes, EMA(21) slope ---
    def _ema_val_slope(period):
        if n_candles < period:
            return None, 0.0
        es = calc_ema(close, period)
        v = es.iloc[-1]
        vp = es.iloc[-2] if len(es) >= 2 else v
        if not ok(v):
            return None, 0.0
        return float(v), (float(v - vp) if ok(vp) else 0.0)

    ema50_val, ema50_slope = _ema_val_slope(50)
    ema200_val, ema200_slope = _ema_val_slope(200)
    ema21_slope = 0.0
    if ok(ema21_val) and n_candles >= 22:
        _e21 = calc_ema(close, 21)
        _e21p = _e21.iloc[-2] if len(_e21) >= 2 else _e21.iloc[-1]
        ema21_slope = float(_e21.iloc[-1] - _e21p) if ok(_e21p) else 0.0

    # --- true SMA200 (None if < 200 bars, per spec) ---
    sma200_true = float(s200) if (has_sma200 and ok(s200)) else None
    sma200_true_slope = sma200_slope if has_sma200 else 0.0

    # --- MACD angle + true percentile + bull phase ---
    macd_slope_angle = math.degrees(math.atan(macd_slope))
    if len(macd_vals) >= 30:
        _stable = macd_vals.iloc[60:] if len(macd_vals) > 60 else macd_vals
        _window = _stable.iloc[-252:] if len(_stable) > 252 else _stable
        macd_pctl_true = round(100.0 * float((_window <= float(cm)).sum()) / len(_window), 1) if len(_window) else 50.0
    else:
        macd_pctl_true = 50.0
    macd_phase_bull = "BULLISH" if (ok(cm) and ok(cms) and cm > cms) else "BEARISH"

    # --- RSI flip (turning up while oversold) ---
    _rsi_series = calc_rsi(close, 14)
    _rsi_prev = _rsi_series.iloc[-2] if len(_rsi_series) >= 2 else r14
    rsi_buy_flip = bool(ok(r14) and ok(_rsi_prev) and float(r14) > float(_rsi_prev) and float(r14) < 50.0)

    # --- spec earlyBuy proxy: MACD hooking up while still below zero ---
    early_buy_spec = bool(macd_accel > 0 and macd_slope > 0 and ok(cm) and cm < 0)

    # --- OBV 5/20 moving averages ---
    obv5_ma = float(ov.rolling(5).mean().iloc[-1]) if len(ov) >= 5 and ok(ov.rolling(5).mean().iloc[-1]) else float(co)
    obv20_ma = float(ov.rolling(20).mean().iloc[-1]) if len(ov) >= 20 and ok(ov.rolling(20).mean().iloc[-1]) else float(co)

    # --- long-term quality metrics ---
    trend_persist = 0.0
    if n_candles >= 200:
        _sfull = calc_sma(close, 200)
        _m = _sfull.notna()
        _cc = close[_m]; _ss = _sfull[_m]
        if len(_cc) >= 1:
            _tail = min(200, len(_cc))
            trend_persist = float((_cc.iloc[-_tail:].values > _ss.iloc[-_tail:].values).mean())

    ema_cascade_ok = bool(ema21_val is not None and ema50_val is not None and ema200_val is not None
                          and ok(ema21_val) and ema21_val > ema50_val > ema200_val)
    ema_cascade_strong = bool(ema_cascade_ok and ema21_slope > 0 and ema50_slope > 0 and ema200_slope > 0)

    _ddwin = close.iloc[-min(252, n_candles):]
    _runmax = _ddwin.cummax()
    max_drawdown = float(((_runmax - _ddwin) / _runmax).max()) if len(_ddwin) else 0.0

    return6m = ((price - float(close.iloc[-126])) / float(close.iloc[-126]) * 100) if n_candles >= 126 and float(close.iloc[-126]) > 0 else 0.0
    return12m = ((price - float(close.iloc[-252])) / float(close.iloc[-252]) * 100) if n_candles >= 252 and float(close.iloc[-252]) > 0 else 0.0

    _rets = close.pct_change().dropna()
    _rt = _rets.iloc[-min(252, len(_rets)):]
    if len(_rt) >= 30:
        _mean = float(_rt.mean()); _std = float(_rt.std(ddof=0))
        sharpe_like = (_mean * 252) / (_std * (252 ** 0.5)) if _std > 0 else 0.0
    else:
        sharpe_like = 0.0

    ind = {
        "price": float(price),
        "sma200Val": sma200_true, "sma200Slope": sma200_true_slope,
        "sma50Val": (float(calc_sma(close, 50).iloc[-1]) if has_sma50 and ok(calc_sma(close, 50).iloc[-1]) else None),
        "ema21Val": (float(ema21_val) if ok(ema21_val) else None), "ema21Slope": ema21_slope,
        "ema50Val": ema50_val, "ema50Slope": ema50_slope,
        "ema200Val": ema200_val, "ema200Slope": ema200_slope,
        "ema21PctDiff": ema21_pct_diff,
        "rsiVal": (float(r14) if ok(r14) else None), "rsiToday": (float(r14) if ok(r14) else 0.0),
        "rsiBuyFlip": rsi_buy_flip,
        "macdSlope": macd_slope, "macdSlopeAngle": macd_slope_angle, "macdAccel": macd_accel,
        "macdPctl": macd_pctl_true, "macdPhaseBull": macd_phase_bull,
        "macdZeroCrossUp": macd_zero_cross_up, "slopeCrossUp": slope_cross_up, "earlyBuy": early_buy_spec,
        "belowMidBand": abm, "touchedLowerBand": tlb,
        "currAdx": float(curr_adx), "bullishTrend": bullish_trend,
        "obvCurrent": float(co), "obv5": obv5_ma, "obv20": obv20_ma,
        "trendPersistencePct": trend_persist,
        "emaCascadeOk": ema_cascade_ok, "emaCascadeStrong": ema_cascade_strong,
        "maxDrawdownPct": max_drawdown,
        "return6mPct": return6m, "return12mPct": return12m, "sharpeLike": sharpe_like,
    }

    pullback_pts, pullback_signal, pullback_rows = scoring.pullback_score(ind)
    momentum_pts, momentum_signal, momentum_rows = scoring.momentum_score(ind)
    comp = scoring.composite(pullback_pts, momentum_pts)

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

    # 252-day price line (like the Android app). Downsampled to ~126 points to
    # keep the results payload light; enough for a smooth sparkline.
    _ph = close.iloc[-min(252, n_candles):]
    _step = max(1, len(_ph) // 126)
    price_history = [round(float(v), 2) for v in _ph.iloc[::_step]]
    if price_history and float(_ph.iloc[-1]) != price_history[-1]:
        price_history.append(round(float(_ph.iloc[-1]), 2))

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
        "price_history":price_history,
        "avg_vol_20":round(avg_vol_20, 0),
        "price_roc3":round(roc3,2),"price_vel":round(price_vel,2),
        "price_accel":round(price_accel,3),"up_days":up_days,
        "support":support,"resistance":resistance,"risk":risk,"reward":reward,"rr_ratio":rr_ratio,
        "atr":round(atr,2),"risk_in_atrs":risk_in_atrs,"position_size":position_size,
        "capital_needed":capital_needed,"potential_profit":potential_profit,"roc_pct":roc_pct,
        "currency": currency_symbol,
        # ── NEW: separated momentum + pullback lenses (replaces old buy_score) ──
        "pullback_score":pullback_pts,"pullback_signal":pullback_signal,"pullback_rows":pullback_rows,
        "momentum_score":momentum_pts,"momentum_signal":momentum_signal,"momentum_rows":momentum_rows,
        "cherry_points":comp["cherryPoints"],"entry_mode":comp["entryMode"],"entry_signal":comp["signal"],
        "sell_score":sell_score,"sell_signal":sell_signal,"sell_breakdown":sell_breakdown,
        "is_buy":comp["cherryPoints"]>=75,"is_moderate_buy":comp["cherryPoints"]>=60,
        "is_momentum":momentum_pts>=60,"is_pullback":pullback_pts>=60,
        "is_sell":sell_score>=65,"is_moderate_sell":sell_score>=45,
        "golden_buy":golden_buy,
    }

# ═══════════════════════════════════════════════════════════════
# SCAN — NIFTY 500 (Yahoo Finance)
# ═══════════════════════════════════════════════════════════════

def run_nifty_scan():
    global scan_results, scan_progress, current_pace
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
    logger.info(f"Starting NIFTY scan: {total} stocks via Yahoo Finance")
    t0 = time.time()
    buys, sells, all_s, errs = [], [], [], []

    portfolio = fetch_portfolio()
    psymbols = {h["symbol"] for h in portfolio}
    consecutive_errors = 0
    retry_queue = []  # stocks that got rate-limited, will retry at end

    for i, inst in enumerate(scan_instrument_list):
        if abort_scan["nifty500"]: break
        sym, yf_sym = inst["symbol"], inst["yf_symbol"]
        name = inst.get("name", sym.replace("-EQ",""))
        clean = sym.replace("-EQ","")

        scan_progress[mkt]["current"] = i + 1
        scan_progress[mkt]["pace"] = round(current_pace, 2)

        if consecutive_errors >= 80:
            for j in range(i, len(scan_instrument_list)):
                retry_queue.append(scan_instrument_list[j])
            logger.warning(f"NIFTY scan: {consecutive_errors} consecutive errors, queued {len(scan_instrument_list) - i} stocks for retry")
            break

        try:
            result = fetch_candle_data(yf_sym)
            if result is RATE_LIMITED:
                consecutive_errors += 1
                retry_queue.append(inst)
                scan_progress[mkt]["errors"] = len(errs)
                current_pace = min(current_pace * PACE_BACKOFF_MULT, PACE_MAX)
                wait = min(3 + consecutive_errors * 0.5, 15)
                time.sleep(wait)
                continue

            if result is None:
                scan_progress[mkt]["skipped"] = scan_progress[mkt].get("skipped", 0) + 1
                consecutive_errors = 0
                current_pace = max(current_pace * PACE_RECOVERY_MULT, PACE_MIN)
                time.sleep(current_pace)  # always pace — even for no-data stocks
                continue

            df = result
            if len(df) < 50:
                scan_progress[mkt]["skipped"] = scan_progress[mkt].get("skipped", 0) + 1
                consecutive_errors = 0; current_pace = max(current_pace * PACE_RECOVERY_MULT, PACE_MIN)
                time.sleep(current_pace)
                continue

            a = analyze_stock(df, currency_symbol="₹")
            if a is None:
                scan_progress[mkt]["skipped"] = scan_progress[mkt].get("skipped", 0) + 1
                consecutive_errors = 0; current_pace = max(current_pace * PACE_RECOVERY_MULT, PACE_MIN)
                time.sleep(current_pace)
                continue

            consecutive_errors = 0
            current_pace = max(current_pace * PACE_RECOVERY_MULT, PACE_MIN)

            indices = get_tags_for(clean, INDEX_TAGS)
            in_portfolio = clean in psymbols
            stock = {"symbol": sym, "name": name, "token": yf_sym,
                     "in_portfolio": in_portfolio, "indices": indices, "market": "nifty500", **a}
            all_s.append(stock)
            scan_progress[mkt]["ok"] = len(all_s)

            if a["is_buy"] or a["is_moderate_buy"]: buys.append(stock)
            if (a["is_sell"] or a["is_moderate_sell"]) and in_portfolio: sells.append(stock)

            if len(all_s) % 10 == 0:
                scan_results[mkt].update({
                    "total_scanned": len(all_s),
                    "buy_signals": sorted(buys, key=lambda x: x["cherry_points"], reverse=True),
                    "sell_signals": sorted(sells, key=lambda x: x["sell_score"], reverse=True),
                    "all_stocks": sorted(all_s, key=lambda x: x["cherry_points"], reverse=True),
                })

            time.sleep(current_pace)
        except Exception as e:
            err_str = str(e)
            errs.append({"symbol": sym, "error": err_str})
            scan_progress[mkt]["errors"] = len(errs)
            if "rate" in err_str.lower() or "timeout" in err_str.lower():
                consecutive_errors += 1
                current_pace = min(current_pace * PACE_BACKOFF_MULT, PACE_MAX)
                retry_queue.append(inst)
                time.sleep(3)
            else:
                consecutive_errors = 0

    # ── Retry phase: process rate-limited stocks with generous pacing ──
    if retry_queue and not abort_scan["nifty500"]:
        already_scanned = {s["token"] for s in all_s}
        retry_queue = [inst for inst in retry_queue if inst["yf_symbol"] not in already_scanned]
        if retry_queue:
            logger.info(f"NIFTY retry phase: {len(retry_queue)} stocks, waiting 10s for rate limit to cool...")
            time.sleep(10)

            retry_pace = 1.5
            retry_errors = 0
            for inst in retry_queue:
                if abort_scan["nifty500"] or retry_errors >= 20: break
                sym, yf_sym = inst["symbol"], inst["yf_symbol"]
                name = inst.get("name", sym.replace("-EQ",""))
                clean = sym.replace("-EQ","")
                scan_progress[mkt]["current"] += 1

                try:
                    result = fetch_candle_data(yf_sym)
                    if result is RATE_LIMITED or result is None:
                        retry_errors += 1
                        errs.append({"symbol": sym, "error": "Rate limited (retry)"})
                        scan_progress[mkt]["errors"] = len(errs)
                        time.sleep(min(5 + retry_errors, 15))
                        continue

                    df = result
                    if len(df) < 50: continue
                    a = analyze_stock(df, currency_symbol="₹")
                    if a is None: continue

                    retry_errors = 0
                    indices = get_tags_for(clean, INDEX_TAGS)
                    in_portfolio = clean in psymbols
                    stock = {"symbol": sym, "name": name, "token": yf_sym,
                             "in_portfolio": in_portfolio, "indices": indices, "market": "nifty500", **a}
                    all_s.append(stock)
                    scan_progress[mkt]["ok"] = len(all_s)
                    if a["is_buy"] or a["is_moderate_buy"]: buys.append(stock)
                    if (a["is_sell"] or a["is_moderate_sell"]) and in_portfolio: sells.append(stock)
                    time.sleep(retry_pace)
                except Exception:
                    retry_errors += 1
                    time.sleep(3)

            logger.info(f"NIFTY retry phase done: recovered {len(all_s)} total stocks")

    elapsed = time.time() - t0
    scan_results[mkt] = {
        "last_scan": datetime.now().isoformat(), "status": "complete",
        "scan_duration_sec": round(elapsed,1), "total_scanned": len(all_s),
        "buy_signals": sorted(buys, key=lambda x: x["cherry_points"], reverse=True),
        "sell_signals": sorted(sells, key=lambda x: x["sell_score"], reverse=True),
        "all_stocks": sorted(all_s, key=lambda x: x["cherry_points"], reverse=True),
        "portfolio_holdings": portfolio, "errors": errs,
    }
    logger.info(f"NIFTY SCAN DONE in {elapsed/60:.1f}min — {len(all_s)} stocks, {len(buys)} buys")
    is_scanning["nifty500"] = False
    threading.Thread(target=enrich_values, args=("nifty500",), daemon=True).start()

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
    scan_progress[mkt] = {"current": 0, "total": total, "ok": 0, "errors": 0, "skipped": 0, "rate_limited": False, "pace": 0.6}
    logger.info(f"Starting NASDAQ scan: {total} stocks")
    t0 = time.time()
    buys, sells, all_s, errs = [], [], [], []
    retry_queue = []
    consecutive_rl = 0

    for i, sym in enumerate(symbols):
        if abort_scan["nasdaq100"]: break
        scan_progress[mkt]["current"] = i + 1

        try:
            df = fetch_us_candle_data(sym)
            if df is US_RATE_LIMITED:
                consecutive_rl += 1
                retry_queue.append(sym)
                scan_progress[mkt]["errors"] = len(errs)
                wait = min(3 + consecutive_rl * 1.0, 15)
                time.sleep(wait)
                continue

            consecutive_rl = 0

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
                    "buy_signals": sorted(buys, key=lambda x: x["cherry_points"], reverse=True),
                    "sell_signals": sorted(sells, key=lambda x: x["sell_score"], reverse=True),
                    "all_stocks": sorted(all_s, key=lambda x: x["cherry_points"], reverse=True),
                })

            time.sleep(0.6)  # pacing for Yahoo
        except Exception as e:
            errs.append({"symbol": sym, "error": str(e)})
            scan_progress[mkt]["errors"] = len(errs)
            retry_queue.append(sym)

    # ── Retry phase for rate-limited NASDAQ stocks ──
    if retry_queue and not abort_scan["nasdaq100"]:
        already = {s["symbol"] for s in all_s}
        retry_queue = [s for s in retry_queue if s not in already]
        if retry_queue:
            logger.info(f"NASDAQ retry phase: {len(retry_queue)} stocks, cooling 10s...")
            time.sleep(10)
            retry_errors = 0
            for sym in retry_queue:
                if abort_scan["nasdaq100"] or retry_errors >= 15: break
                try:
                    df = fetch_us_candle_data(sym)
                    if df is US_RATE_LIMITED or df is None or len(df) < 50:
                        retry_errors += 1
                        time.sleep(min(5 + retry_errors, 15))
                        continue
                    a = analyze_stock(df, currency_symbol="$", min_avg_volume=50000)
                    if a is None: continue
                    retry_errors = 0
                    stock = {"symbol": sym, "name": sym, "token": sym,
                             "in_portfolio": False, "indices": ["NASDAQ 100"], "market": "nasdaq100", **a}
                    all_s.append(stock)
                    scan_progress[mkt]["ok"] = len(all_s)
                    if a["is_buy"] or a["is_moderate_buy"]: buys.append(stock)
                    if a["is_sell"] or a["is_moderate_sell"]: sells.append(stock)
                    time.sleep(1.5)
                except Exception:
                    retry_errors += 1
                    time.sleep(3)
            logger.info(f"NASDAQ retry done: {len(all_s)} total stocks")

    elapsed = time.time() - t0
    scan_results[mkt] = {
        "last_scan": datetime.now().isoformat(), "status": "complete",
        "scan_duration_sec": round(elapsed,1), "total_scanned": len(all_s),
        "buy_signals": sorted(buys, key=lambda x: x["cherry_points"], reverse=True),
        "sell_signals": sorted(sells, key=lambda x: x["sell_score"], reverse=True),
        "all_stocks": sorted(all_s, key=lambda x: x["cherry_points"], reverse=True),
        "portfolio_holdings": [], "errors": errs,
    }
    logger.info(f"NASDAQ SCAN DONE in {elapsed/60:.1f}min — {len(all_s)} stocks, {len(buys)} buys")
    is_scanning["nasdaq100"] = False
    threading.Thread(target=enrich_values, args=("nasdaq100",), daemon=True).start()

# ═══════════════════════════════════════════════════════════════
# COMBINED SCAN (parallel)
# ═══════════════════════════════════════════════════════════════

def run_full_scan(markets=None):
    """Launch scans for requested markets. Each market runs independently."""
    if markets is None:
        markets = ["nifty500", "nasdaq100"]

    threads = []
    if "nifty500" in markets and not is_scanning["nifty500"]:
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

# Set SKIP_AUTOSCAN=1 to bring the dashboard up WITHOUT auto-scanning on launch
# (useful to get a working URL if a scan is crashing — trigger scans manually).
SKIP_AUTOSCAN = os.getenv("SKIP_AUTOSCAN", "").strip() in ("1", "true", "True")

def scheduler():
    if SKIP_AUTOSCAN:
        logger.info("SKIP_AUTOSCAN set — scheduler idle; use the Scan button to scan manually.")
        return
    # give the web server a moment to come up and be reachable before we launch
    # into a heavy scan
    time.sleep(5)
    while True:
        now = datetime.now()
        wd = now.weekday() < 5
        # Indian market hours
        ao = (now.hour==9 and now.minute>=15) or now.hour>=10
        bc = now.hour<15 or (now.hour==15 and now.minute<=30)
        try:
            if wd and ao and bc:
                run_full_scan(["nifty500"])
            # US market scan at 8 PM IST (US market open ~7 PM IST)
            if wd and now.hour == 20 and now.minute < 20:
                run_full_scan(["nasdaq100"])
        except Exception:
            logger.exception("Scheduled scan raised — continuing")
        time.sleep(SCAN_INTERVAL)

# ═══════════════════════════════════════════════════════════════
# VALUE (fundamentals) + AI (news + LLM) — all ON-DEMAND
# These are intentionally NOT run during bulk scans: fetching fundamentals
# and news for 500+ stocks would blow the Yahoo rate limits and cost tokens.
# ═══════════════════════════════════════════════════════════════

# marketLabel / currency / analyst persona per scan universe (spec §6)
MARKET_CONTEXT = {
    "nifty500":  {"label": "NSE India",  "currency": "₹", "persona": "Indian equity investors"},
    "nasdaq100": {"label": "US Markets", "currency": "$", "persona": "US equity investors"},
}

def _market_context(market):
    return MARKET_CONTEXT.get(market, MARKET_CONTEXT["nasdaq100"])

def _find_scan_stock(symbol, market=None):
    """Locate a previously-scanned stock (for its price/name/ownership/yf ticker)."""
    markets = [market] if market else list(scan_results.keys())
    for mkt in markets:
        for s in scan_results.get(mkt, {}).get("all_stocks", []):
            if symbol in (s.get("symbol"), s.get("name"), s.get("token"),
                          str(s.get("symbol", "")).replace("-EQ", "")):
                return s, mkt
    return None, market

def fetch_fundamentals(yf_ticker):
    """
    Fetch Yahoo fundamentals via yfinance and normalize to scoring.value_score's
    contract. Returns (fundamentals_dict, price, sector) or (None, None, None).
    """
    try:
        import yfinance as yf
        info = yf.Ticker(yf_ticker).info or {}
    except Exception as e:
        logger.warning(f"Fundamentals fetch failed for {yf_ticker}: {e}")
        return None, None, None

    if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
        return None, None, None

    def pct(v):   # yfinance fraction (0.10) → percent (10.0)
        return v * 100 if isinstance(v, (int, float)) else None
    def div_pct(v):
        if not isinstance(v, (int, float)):
            return None
        return v * 100 if v < 1 else v  # yfinance versions differ; normalize to percent
    def ratio_de(v):
        return v / 100.0 if isinstance(v, (int, float)) else None  # yfinance D/E is a percent

    f = {
        "trailingPE": info.get("trailingPE"),
        "forwardPE": info.get("forwardPE"),
        "priceToBook": info.get("priceToBook"),
        "enterpriseToEbitda": info.get("enterpriseToEbitda"),
        "debtToEquity": ratio_de(info.get("debtToEquity")),
        "returnOnEquity": info.get("returnOnEquity"),          # fraction (scoring ×100)
        "operatingCashflow": info.get("operatingCashflow"),
        "freeCashflow": info.get("freeCashflow"),
        "netIncome": info.get("netIncomeToCommon"),
        "totalRevenue": info.get("totalRevenue"),
        "totalCash": info.get("totalCash"),
        "totalDebt": info.get("totalDebt"),
        "currentRatio": info.get("currentRatio"),
        "revenueGrowth": pct(info.get("revenueGrowth")),
        "earningsGrowth": pct(info.get("earningsGrowth")),
        "grossMargins": pct(info.get("grossMargins")),
        "operatingMargins": pct(info.get("operatingMargins")),
        "profitMargins": pct(info.get("profitMargins")),
        "dividendYield": div_pct(info.get("dividendYield")),
        "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
    }
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    return f, price, info.get("sector")

# ── Value-score enrichment: compute value_score for every scanned stock in a
# paced background thread (fundamentals are too rate-limit-heavy for the scan
# loop itself). Mutates the stored stock dicts in place so /api/results picks
# them up on the next poll. ──
value_enrich = {
    "nifty500":  {"running": False, "done": 0, "total": 0, "ts": None},
    "nasdaq100": {"running": False, "done": 0, "total": 0, "ts": None},
}
_value_cache = {}          # yf_ticker -> (score, rating, sector, ts)
VALUE_CACHE_TTL = 6 * 3600  # fundamentals barely move intraday

def enrich_values(mkt, force=False):
    if mkt not in value_enrich or value_enrich[mkt]["running"]:
        return
    stocks = scan_results.get(mkt, {}).get("all_stocks", [])
    if not stocks:
        return
    value_enrich[mkt].update({"running": True, "done": 0, "total": len(stocks)})
    logger.info(f"Value enrichment start ({mkt}): {len(stocks)} stocks")
    try:
        for s in stocks:
            if abort_scan.get(mkt):
                break
            yf_ticker = s.get("token") or s.get("symbol")
            if not force and isinstance(s.get("value_score"), int):
                value_enrich[mkt]["done"] += 1
                continue
            cached = _value_cache.get(yf_ticker)
            if cached and (time.time() - cached[3]) < VALUE_CACHE_TTL:
                s["value_score"], s["value_rating"], s["value_sector"] = cached[0], cached[1], cached[2]
                value_enrich[mkt]["done"] += 1
                continue
            try:
                f, price, sector = fetch_fundamentals(yf_ticker)
                if f is None:
                    s["value_score"] = None; s["value_rating"] = "N/A"
                else:
                    score, rating, _rows = scoring.value_score(f, price or s.get("price") or 0.0)
                    s["value_score"] = score; s["value_rating"] = rating; s["value_sector"] = sector
                    _value_cache[yf_ticker] = (score, rating, sector, time.time())
            except Exception as e:
                logger.debug(f"Value enrich error {yf_ticker}: {e}")
                s["value_score"] = None; s["value_rating"] = "N/A"
            value_enrich[mkt]["done"] += 1
            time.sleep(0.6)  # pace to stay under Yahoo rate limits
    finally:
        value_enrich[mkt]["running"] = False
        value_enrich[mkt]["ts"] = datetime.now().isoformat()
        logger.info(f"Value enrichment done ({mkt}): {value_enrich[mkt]['done']} processed")

def _strip_exchange_suffix(ticker):
    for suf in (".NS", ".BO", ".DE", ".NSE", ".BSE"):
        if ticker.endswith(suf):
            return ticker[: -len(suf)]
    return ticker

def fetch_news_headlines(ticker, name=None, market=None, limit=12, days=30):
    """
    Pull recent headlines from Google News RSS (no API key). Returns a list of
    {"title","pubDate"} dicts, most-recent-first, from the last `days` days.
    """
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime
    base = _strip_exchange_suffix(ticker)
    q = base
    if name and name.upper() != base.upper():
        q = f"{base} {name}"
    q = f"{q} stock"
    url = "https://news.google.com/rss/search"
    params = {"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    try:
        resp = requests.get(url, params=params, timeout=12,
                            headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return []
        root = ET.fromstring(resp.content)
        cutoff = datetime.now().astimezone() - timedelta(days=days)
        items = []
        for item in root.iter("item"):
            title = item.findtext("title") or ""
            pub = item.findtext("pubDate") or ""
            try:
                dt = parsedate_to_datetime(pub)
                if dt is None:
                    continue
                if dt.tzinfo is None:
                    dt = dt.astimezone()
                if dt < cutoff:
                    continue
            except Exception:
                continue
            items.append({"title": title.strip(), "pubDate": pub, "_dt": dt})
        items.sort(key=lambda x: x["_dt"], reverse=True)
        return [{"title": i["title"], "pubDate": i["pubDate"]} for i in items[:limit]]
    except Exception as e:
        logger.debug(f"News fetch error {ticker}: {e}")
        return []

def call_llm(system, user, max_tokens=500):
    """Dispatch to the configured AI provider. Returns text, or raises RuntimeError."""
    if AI_PROVIDER == "openai":
        return _call_openai(system, user, max_tokens)
    return _call_anthropic(system, user, max_tokens)

def _call_openai(system, user, max_tokens=500):
    """Call the OpenAI Chat Completions API (ChatGPT)."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not configured — set it in .env to enable AI features")
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.2,
            },
            timeout=60,
        )
        data = resp.json()
        if resp.status_code != 200:
            raise RuntimeError((data.get("error") or {}).get("message", f"HTTP {resp.status_code}"))
        return (data["choices"][0]["message"]["content"] or "").strip()
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"AI request failed: {e}")

def _call_anthropic(system, user, max_tokens=500):
    """Call the Anthropic Messages API (Claude)."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not configured — set it in .env to enable AI features")
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": AI_MODEL,
                "max_tokens": max_tokens,
                "temperature": 0.2,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            timeout=60,
        )
        data = resp.json()
        if resp.status_code != 200:
            raise RuntimeError(data.get("error", {}).get("message", f"HTTP {resp.status_code}"))
        parts = data.get("content", [])
        return "".join(p.get("text", "") for p in parts if p.get("type") == "text").strip()
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"AI request failed: {e}")

def _rsi_status(rsi):
    if rsi < 30: return "oversold — may bounce soon"
    if rsi < 50: return "mid-range — pullback zone ideal"
    if rsi < 70: return "near exhaustion — caution on further bounce"
    return "overbought — unusual for a pullback"

def _adx_status(adx):
    if adx > 25: return "trending"
    if adx > 15: return "emerging trend"
    return "choppy"

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
    instruments_loaded = len(scan_instrument_list) > 0
    return jsonify({
        "connected": instruments_loaded,
        "instruments_scan": len(scan_instrument_list),
        "last_scan": scan_results.get("nifty500", {}).get("last_scan"),
        "scan_status": "scanning" if any(is_scanning.values()) else "idle",
        "is_scanning": any(is_scanning.values()),
        "is_scanning_per_market": is_scanning,
        "missing": [],
        "zerodha_connected": zerodha_access_token is not None,
        "zerodha_user": zerodha_user.get("user_name","") if zerodha_user else "",
        "zerodha_configured": bool(ZERODHA_API_KEY and ZERODHA_API_SECRET),
        "login_url": f"https://kite.zerodha.com/connect/login?v=3&api_key={ZERODHA_API_KEY}" if ZERODHA_API_KEY else "",
        "markets": {
            "nifty500": {"available": True, "stocks": len(scan_instrument_list),
                         "last_scan": scan_results["nifty500"].get("last_scan"),
                         "status": scan_results["nifty500"].get("status", "not_started")},
            "nasdaq100": {"available": True, "stocks": len(get_nasdaq100_symbols()),
                          "last_scan": scan_results["nasdaq100"].get("last_scan"),
                          "status": scan_results["nasdaq100"].get("status", "not_started")},
        }
    })

@app.route("/api/reconnect", methods=["POST"])
def api_reconnect():
    load_instrument_list()
    return jsonify({"status": "ready", "instruments_scan": len(scan_instrument_list)})

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
    if len(scan_instrument_list) == 0:
        load_instrument_list()
    return jsonify({"status": "reset", "connected": len(scan_instrument_list) > 0})

@app.route("/api/scan", methods=["POST"])
def api_trigger_scan():
    body = request.get_json() or {}
    markets = body.get("markets", ["nifty500", "nasdaq100"])
    if isinstance(markets, str):
        markets = [markets]

    if "nifty500" in markets:
        if len(scan_instrument_list) == 0:
            load_instrument_list()

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

# ── Chart cache: avoids Yahoo rate limits during scans ──
_chart_cache = {}  # key: "symbol|resolution|period" -> {"data": ..., "ts": timestamp}
CHART_CACHE_TTL = 900  # 15 minutes

@app.route("/api/index_chart")
def api_index_chart():
    """Fetch index chart data with caching to avoid Yahoo rate limits during scans."""
    index_symbol = request.args.get("index", "^NSEI")
    resolution = request.args.get("resolution", "1d")
    period = request.args.get("period", "1y")
    cache_key = f"{index_symbol}|{resolution}|{period}"

    # Return cached data if fresh enough
    cached = _chart_cache.get(cache_key)
    if cached and (time.time() - cached["ts"]) < CHART_CACHE_TTL:
        logger.info(f"Index chart cache hit: {index_symbol} (age {int(time.time() - cached['ts'])}s)")
        return jsonify(cached["data"])

    # Check if any scan is actively running — if so, don't hit Yahoo (will get rate limited)
    if any(is_scanning.values()):
        if cached:
            # Return stale cache rather than nothing
            logger.info(f"Index chart: scan running, returning stale cache for {index_symbol}")
            return jsonify(cached["data"])
        else:
            logger.warning(f"Index chart: scan running, no cache for {index_symbol}")
            return jsonify({"error": "Charts unavailable during scan. Will load after scan completes."}), 503

    logger.info(f"Index chart request: {index_symbol} res={resolution} period={period}")
    for attempt in range(2):
        try:
            data = fetch_index_chart_data(index_symbol, resolution, period)
            if data is not None:
                logger.info(f"Index chart OK: {index_symbol} — {data.get('data_points', 0)} points")
                _chart_cache[cache_key] = {"data": data, "ts": time.time()}
                return jsonify(data)
            logger.warning(f"Index chart returned None for {index_symbol} (attempt {attempt+1})")
            if attempt == 0:
                time.sleep(1)
        except Exception as e:
            logger.error(f"Index chart error (attempt {attempt+1}) for {index_symbol}: {e}")
            if attempt == 0:
                time.sleep(1)
    return jsonify({"error": "Could not fetch index data. Try again when no scan is running."}), 500

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

# ── AI / VALUE endpoints (on-demand) ──

@app.route("/api/ai/config")
def api_ai_config():
    """Frontend uses this to know whether to show AI buttons."""
    return jsonify({"ai_enabled": ai_is_enabled(), "provider": AI_PROVIDER, "model": ai_active_model()})

@app.route("/api/value")
def api_value():
    """Compute the fundamental Value score (spec §4) for one symbol, on demand."""
    symbol = request.args.get("symbol", "").strip()
    market = request.args.get("market", None)
    if not symbol:
        return jsonify({"error": "symbol required"}), 400
    stock, mkt = _find_scan_stock(symbol, market)
    yf_ticker = (stock or {}).get("token") or symbol
    f, price, sector = fetch_fundamentals(yf_ticker)
    if f is None:
        return jsonify({"error": f"No fundamentals available for {symbol}"}), 404
    if not price and stock:
        price = stock.get("price")
    score, rating, rows = scoring.value_score(f, price or 0.0)
    return jsonify({"symbol": symbol, "market": mkt, "sector": sector,
                    "score": score, "rating": rating, "rows": rows, "price": price})

@app.route("/api/value/enrich", methods=["POST"])
def api_value_enrich():
    """Kick off background value-score enrichment for all stocks in a market."""
    body = request.get_json(silent=True) or {}
    mkt = body.get("market") or request.args.get("market", "nifty500")
    if mkt not in value_enrich:
        return jsonify({"error": "unknown market"}), 400
    if value_enrich[mkt]["running"]:
        return jsonify({"status": "already_running", **value_enrich[mkt]})
    force = bool(body.get("force"))
    threading.Thread(target=enrich_values, args=(mkt, force), daemon=True).start()
    return jsonify({"status": "started", "market": mkt,
                    "total": len(scan_results.get(mkt, {}).get("all_stocks", []))})

@app.route("/api/value/status")
def api_value_status():
    """Progress of value enrichment per market (for the discovery table)."""
    return jsonify(value_enrich)

@app.route("/api/ai/pullback", methods=["POST"])
def api_ai_pullback():
    """AI Pullback Analysis (spec §5): TEMPORARY / STRUCTURAL / UNCERTAIN."""
    body = request.get_json() or {}
    symbol = (body.get("symbol") or "").strip()
    market = body.get("market")
    if not symbol:
        return jsonify({"error": "symbol required"}), 400
    stock, mkt = _find_scan_stock(symbol, market)
    if not stock:
        return jsonify({"error": f"{symbol} not found in scan results — run a scan first"}), 404
    ctx = _market_context(mkt)
    name = stock.get("name", symbol)
    ticker = _strip_exchange_suffix(stock.get("token") or symbol)
    headlines = fetch_news_headlines(stock.get("token") or symbol, name, mkt)
    if not headlines:
        return jsonify({"error": f"No recent news found for {symbol}"}), 404

    price = stock.get("price") or 0.0
    rsi = stock.get("rsi") or 0.0
    adx = stock.get("adx") or 0.0
    support = stock.get("support") or 0.0
    pct_above_support = ((price - support) / support * 100) if support else 0.0
    news_block = "\n".join(f"[{i+1}] {h['title']}" for i, h in enumerate(headlines))

    system = (
        "You are a stock market analyst. Analyze a deep pullback critically and independently.\n"
        "Form your own opinion based PRIMARILY on recent news (headlines appear in chronological order, most recent first).\n"
        "Do NOT assume the pullback is automatically a buying opportunity — critically assess both risks and opportunities.\n\n"
        "Evaluate whether this pullback is:\n"
        "• TEMPORARY: A healthy correction in an uptrend; news suggests a one-off event or overreaction\n"
        "• STRUCTURAL: Signs of fundamental weakness; news suggests deteriorating business/sector conditions\n"
        "• UNCERTAIN: Conflicting signals; need more confirmation before committing\n\n"
        "Prioritize recent news over older stories. Be concise (3-4 sentences max). If you're unsure, answer UNCERTAIN rather than guessing."
    )
    user = (
        f"Stock: {name} ({ticker}) ({ctx['label']})\n"
        f"Current price: {ctx['currency']}{price:.2f}\n"
        f"EMA(21) distance: {stock.get('ema21_pct_diff', 0):.1f}% (deep pullback trigger)\n"
        f"MACD Phase: {stock.get('macd_phase', 'N/A')}\n\n"
        f"─ Technical Context ─\n"
        f"RSI = {rsi:.0f} ({_rsi_status(rsi)})\n"
        f"ADX = {adx:.1f} ({_adx_status(adx)})\n"
        f"Price is {pct_above_support:.1f}% above support ({ctx['currency']}{support:.2f})\n\n"
        f"─ Recent News Headlines (last 30 days) ─\n{news_block}\n\n"
        "Analyze critically:\n"
        "• Based on the news, what is the PRIMARY driver of this pullback?\n"
        "• Is this a temporary market overreaction or structural weakness?\n"
        "• What would need to happen for this to be a BUY vs AVOID?\n"
        "• If uncertain, say UNCERTAIN rather than guessing.\n\n"
        "Provide your verdict: TEMPORARY, STRUCTURAL, or UNCERTAIN."
    )
    try:
        text = call_llm(system, user, max_tokens=500)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502
    return jsonify({"symbol": symbol, "analysis": text, "headlines": headlines})

@app.route("/api/ai/outlook", methods=["POST"])
def api_ai_outlook():
    """AI Stock Outlook (spec §6): structured outlook + verdict."""
    body = request.get_json() or {}
    symbol = (body.get("symbol") or "").strip()
    market = body.get("market")
    if not symbol:
        return jsonify({"error": "symbol required"}), 400
    stock, mkt = _find_scan_stock(symbol, market)
    ctx = _market_context(mkt)
    name = (stock or {}).get("name", symbol)
    ticker = _strip_exchange_suffix((stock or {}).get("token") or symbol)
    owned = bool(body.get("owned", (stock or {}).get("in_portfolio", False)))
    price = (stock or {}).get("price") or 0.0

    headlines = fetch_news_headlines((stock or {}).get("token") or symbol, name, mkt)

    if owned:
        allowed = "STRONG BUY / BUY / HOLD / REDUCE / SELL"
        hold_rule = "HOLD is allowed because the user currently owns this stock."
    else:
        allowed = "STRONG BUY / BUY / WATCH / AVOID"
        hold_rule = ("HOLD is NOT a valid verdict — the user does NOT own this stock, so you must "
                     "commit to a directional call (BUY / WATCH / AVOID). Do not output HOLD under any circumstances.")
    system = (
        f"You are a stock market analyst providing outlook summaries for {ctx['persona']}.\n"
        "Form your own independent opinion based PRIMARILY on recent news and publicly known fundamentals.\n"
        "Headlines are listed in chronological order (most recent first) — prioritize those heavily over older news.\n"
        "Do NOT assume any prior bullish or bearish bias — you are given current price, ownership status, and headlines only. Critically assess both risks and opportunities.\n"
        "Do not infer or mention app technical indicators unless they are explicitly present in the news headlines.\n\n"
        "Give a structured response:\n"
        "1. SHORT TERM (1-4 weeks): News-driven outlook\n"
        "2. LONG TERM (3-12 months): Fundamental narrative from recent news and publicly known context\n"
        "3. KEY RISKS: 1-2 bullet points from recent developments\n"
        f"4. VERDICT: One of: {allowed}\n\n"
        f"{hold_rule}\n\n"
        "Keep each section to 2-3 sentences max. Be specific about price levels when possible."
    )
    if headlines:
        news_block = "\n".join(f"[{i+1}] {h['title']}" for i, h in enumerate(headlines))
    else:
        news_block = ("(No recent news available via Google News RSS — rely on general market knowledge of this "
                      "company, and note the absence of news in your response)")
    user = (
        f"Stock: {name} ({ticker}) ({ctx['label']})\n"
        f"Current Price: {ctx['currency']}{price:.2f}\n"
        f"User currently owns this: {'YES' if owned else 'NO'}\n\n"
        f"── Recent News Headlines (prioritize items from the last 30 days) ──\n{news_block}\n\n"
        "Based on the above, provide the structured outlook. Cite headline numbers inline where relevant. "
        f"{'Remember: HOLD is NOT a valid verdict here. ' if not owned else ''}Do not reference any app-generated scoring."
    )
    try:
        text = call_llm(system, user, max_tokens=500)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502
    return jsonify({"symbol": symbol, "outlook": text, "owned": owned, "headlines": headlines})

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
    # Load NIFTY 500 universe from index_data (no broker credentials needed)
    load_instrument_list()
    if len(scan_instrument_list) > 0:
        logger.info(f"Instruments loaded: {len(scan_instrument_list)} stocks (Yahoo Finance)")
    else:
        logger.warning("Instrument load failed — NIFTY scan may be limited.")

    if zerodha_access_token and ZERODHA_API_KEY:
        if _verify_zerodha_token():
            logger.info("Zerodha token valid.")

    threading.Thread(target=scheduler, daemon=True).start()
    logger.info(f"Ready. NIFTY: {len(scan_instrument_list)} stocks, NASDAQ: {len(get_nasdaq100_symbols())} stocks")

    # Pre-fetch index charts into cache so they're available during scans
    def _prefetch_charts():
        charts_to_prefetch = [
            ("^NSEI", "1d", "1y"), ("^NSEI", "1d", "3mo"), ("^NSEI", "1d", "6mo"), ("^NSEI", "1d", "1mo"),
            ("^NSEBANK", "1d", "1y"), ("^BSESN", "1d", "1y"),
            ("^IXIC", "1d", "1y"), ("^IXIC", "1d", "3mo"),
        ]
        for sym, res, per in charts_to_prefetch:
            try:
                data = fetch_index_chart_data(sym, res, per)
                if data:
                    cache_key = f"{sym}|{res}|{per}"
                    _chart_cache[cache_key] = {"data": data, "ts": time.time()}
                    logger.info(f"Pre-cached chart: {sym} {per}")
                time.sleep(0.3)  # gentle pacing to avoid rate limit
            except Exception as e:
                logger.debug(f"Pre-cache failed for {sym}: {e}")
        logger.info(f"Chart pre-cache done: {len(_chart_cache)} charts cached")
    threading.Thread(target=_prefetch_charts, daemon=True).start()

    return True
