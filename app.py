"""
SignalScope v3 — Scans ALL NSE equities (~2000 stocks)
Daily candles, 6 indicators, 100-point weighted scoring.
Stocks tagged with known indices (NIFTY 50/100/200, BSE 100, Midcap 150)
or "OTHER NSE" for everything else.
"""

import os, time, threading, logging
from datetime import datetime, timedelta
import numpy as np, pandas as pd, pyotp, requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from SmartApi import SmartConnect
from index_data import build_index_map, get_indices_for, INDEX_NAMES, ALL_KNOWN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

API_KEY    = os.getenv("ANGEL_API_KEY", "")
CLIENT_ID  = os.getenv("ANGEL_CLIENT_ID", "")
PASSWORD   = os.getenv("ANGEL_PASSWORD", "")
TOTP_TOKEN = os.getenv("ANGEL_TOTP_TOKEN", "")
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "900"))

# SCAN_MODE: "top500" (fast ~5min) or "full" (all NSE ~15min)
scan_mode = os.getenv("SCAN_MODE", "top500").lower()

INDEX_MAP = build_index_map()

smart_api = None
refresh_token = None
instrument_list = []       # full list from Angel One
scan_instrument_list = []  # filtered list based on scan_mode
is_scanning = False
credentials_ok = False
last_login_attempt = 0     # timestamp of last login attempt
login_backoff = 1          # exponential backoff seconds

# Live scan progress (read by frontend)
scan_progress = {"current": 0, "total": 0, "ok": 0, "errors": 0, "rate_limited": False}

scan_results = {
    "last_scan": None, "status": "not_started", "total_scanned": 0,
    "buy_signals": [], "sell_signals": [], "all_stocks": [],
    "portfolio_holdings": [], "errors": [],
    "index_counts": {n: 0 for n in INDEX_NAMES},
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
        login_backoff = 1  # reset backoff on success
        logger.info("Logged in to Angel One")
        return True
    except Exception as e:
        logger.error(f"Login error: {e}")
        credentials_ok = False
        return False

def ensure_session():
    """Re-login if needed, with exponential backoff to avoid rate limit spiral."""
    global last_login_attempt, login_backoff, scan_progress
    if smart_api is None:
        return create_session()
    try:
        smart_api.getProfile(refresh_token)
        return True
    except:
        # Check cooldown — don't spam login attempts
        now = time.time()
        if now - last_login_attempt < login_backoff:
            return False  # still cooling down
        last_login_attempt = now
        result = create_session()
        if not result:
            # Exponential backoff: 1s, 2s, 4s, 8s, 16s, max 60s
            login_backoff = min(login_backoff * 2, 60)
            scan_progress["rate_limited"] = True
            logger.warning(f"Login failed, backing off {login_backoff}s")
        else:
            login_backoff = 1
            scan_progress["rate_limited"] = False
        return result

# ═══════════════════════════════════════════════════════════════
# INSTRUMENTS — ALL NSE EQUITIES
# ═══════════════════════════════════════════════════════════════

def fetch_instrument_list():
    """Download ALL NSE equities, then apply scan_mode filter."""
    global instrument_list, scan_instrument_list
    try:
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        logger.info("Downloading instrument list...")
        resp = requests.get(url, timeout=120)
        all_inst = resp.json()

        # Full NSE equity list (always kept for switching modes without re-download)
        instrument_list = [
            i for i in all_inst
            if i.get("exch_seg") == "NSE" and i.get("symbol", "").endswith("-EQ")
        ]
        logger.info(f"Total NSE equities: {len(instrument_list)}")

        apply_scan_mode()
        return True
    except Exception as e:
        logger.error(f"Instrument fetch failed: {e}")
        return False


def apply_scan_mode():
    """Filter instrument_list based on scan_mode. No re-download needed."""
    global scan_instrument_list
    if scan_mode == "full":
        scan_instrument_list = instrument_list
        logger.info(f"Scan mode: FULL — {len(scan_instrument_list)} stocks")
    else:
        # Top 500: use all known index members (~300) + next ~200 by sorting
        # Angel One's instrument list is roughly ordered by market cap
        known = ALL_KNOWN
        known_list = [i for i in instrument_list if i["symbol"].replace("-EQ","") in known]
        other_list = [i for i in instrument_list if i["symbol"].replace("-EQ","") not in known]
        # Take first ~200 unknowns (roughly next by market cap in Angel's list)
        extra = other_list[:max(0, 500 - len(known_list))]
        scan_instrument_list = known_list + extra
        logger.info(f"Scan mode: TOP500 — {len(scan_instrument_list)} stocks ({len(known_list)} indexed + {len(extra)} other)")

# ═══════════════════════════════════════════════════════════════
# DATA FETCH
# ═══════════════════════════════════════════════════════════════

def fetch_candle_data(symbol_token):
    """Fetch 365 days of daily candles."""
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

def calc_adx(high, low, close, p=14):
    pdf = pd.DataFrame({"ph": high.diff(), "pl": low.shift(1) - low})
    pdf["plus_dm"] = pdf.apply(lambda r: r["ph"] if r["ph"] > r["pl"] and r["ph"] > 0 else 0, axis=1)
    pdf["minus_dm"] = pdf.apply(lambda r: r["pl"] if r["pl"] > r["ph"] and r["pl"] > 0 else 0, axis=1)
    tr1 = high - low; tr2 = (high - close.shift(1)).abs(); tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=p, adjust=False).mean()
    plus_di = 100 * (pdf["plus_dm"].ewm(span=p, adjust=False).mean() / atr)
    minus_di = 100 * (pdf["minus_dm"].ewm(span=p, adjust=False).mean() / atr)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    return dx.ewm(span=p, adjust=False).mean()

def find_support_resistance(high, low, close, lookback=60):
    price = close.iloc[-1]
    rh, rl, rc = high.iloc[-lookback:], low.iloc[-lookback:], close.iloc[-lookback:]
    swing_highs, swing_lows = [], []
    for i in range(2, len(rh)-2):
        if rh.iloc[i]>rh.iloc[i-1] and rh.iloc[i]>rh.iloc[i-2] and rh.iloc[i]>rh.iloc[i+1] and rh.iloc[i]>=rh.iloc[i+2]:
            swing_highs.append(rh.iloc[i])
        if rl.iloc[i]<rl.iloc[i-1] and rl.iloc[i]<rl.iloc[i-2] and rl.iloc[i]<rl.iloc[i+1] and rl.iloc[i]<=rl.iloc[i+2]:
            swing_lows.append(rl.iloc[i])
    pivot = (rh.iloc[-1]+rl.iloc[-1]+rc.iloc[-1])/3
    r1, s1 = 2*pivot-rl.iloc[-1], 2*pivot-rh.iloc[-1]
    r2, s2 = pivot+(rh.iloc[-1]-rl.iloc[-1]), pivot-(rh.iloc[-1]-rl.iloc[-1])
    res = sorted(set([r1,r2]+[h for h in swing_highs if h>price]))
    sup = sorted(set([s1,s2]+[l for l in swing_lows if l<price]), reverse=True)
    support = next((s for s in sup if s<price), rl.min())
    resistance = next((r for r in res if r>price), rh.max())
    return round(float(support),2), round(float(resistance),2)

# ═══════════════════════════════════════════════════════════════
# ANALYZE — 100pt WEIGHTED SCORING
# ═══════════════════════════════════════════════════════════════

def analyze_stock(df):
    if df is None or len(df) < 50: return None
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    n_candles = len(df)
    price = close.iloc[-1]
    ok = lambda v: not pd.isna(v)
    sf = lambda v, d=2: round(float(v),d) if ok(v) else None

    # SMA(200) — only calculate if we have enough data
    s200 = calc_sma(close,200).iloc[-1] if n_candles >= 200 else np.nan

    r14 = calc_rsi(close,14).iloc[-1]
    bm,bu,bl = calc_bb(close,20,2)
    ml,ms,mh = calc_macd(close)
    ov = calc_obv(close,vol)
    adx_s = calc_adx(high,low,close,14)

    cbm,cbu,cbl = bm.iloc[-1],bu.iloc[-1],bl.iloc[-1]
    cm,cms,cmh = ml.iloc[-1],ms.iloc[-1],mh.iloc[-1]

    # ── MACD Derivatives (d/dt and d²/dt²) ──
    # First derivative: slope of MACD line (velocity of momentum)
    cm_prev = ml.iloc[-2] if len(ml)>=2 else cm
    cm_prev2 = ml.iloc[-3] if len(ml)>=3 else cm_prev
    macd_slope = float(cm - cm_prev) if ok(cm) and ok(cm_prev) else 0      # d(MACD)/dt
    macd_slope_prev = float(cm_prev - cm_prev2) if ok(cm_prev) and ok(cm_prev2) else 0
    # Second derivative: rate of change of slope (acceleration of momentum)
    macd_accel = macd_slope - macd_slope_prev  # d²(MACD)/dt²

    # ── Momentum Phase Detection ──
    # Phase 1 (EARLY BUY):  slope < 0 but accel > 0  → decline slowing, about to flip
    # Phase 2 (BUY CONFIRM): slope crosses 0 from below (was neg, now pos)
    # Phase 3 (EARLY SELL):  slope > 0 but accel < 0  → rise slowing, about to flip
    # Phase 4 (SELL CONFIRM): slope crosses 0 from above (was pos, now neg)
    slope_cross_up = bool(macd_slope > 0 and macd_slope_prev <= 0)   # just flipped positive
    slope_cross_dn = bool(macd_slope < 0 and macd_slope_prev >= 0)   # just flipped negative
    early_buy = bool(macd_slope < 0 and macd_accel > 0)              # decline decelerating
    early_sell = bool(macd_slope > 0 and macd_accel < 0)             # rise decelerating

    if slope_cross_up:
        macd_phase = "BUY FLIP"
    elif early_buy:
        macd_phase = "EARLY BUY"
    elif slope_cross_dn:
        macd_phase = "SELL FLIP"
    elif early_sell:
        macd_phase = "EARLY SELL"
    elif macd_slope > 0 and macd_accel >= 0:
        macd_phase = "BULLISH"
    elif macd_slope < 0 and macd_accel <= 0:
        macd_phase = "BEARISH"
    else:
        macd_phase = "NEUTRAL"
    co = ov.iloc[-1]
    co5 = ov.iloc[-5] if len(ov)>=5 else ov.iloc[0]
    co20 = ov.iloc[-20] if len(ov)>=20 else ov.iloc[0]
    curr_adx = adx_s.iloc[-1] if len(adx_s)>0 and ok(adx_s.iloc[-1]) else 0
    tlb = any(close.iloc[i]<=bl.iloc[i] for i in range(-5,0) if ok(bl.iloc[i]))
    abm = bool(price<=cbm) if ok(cbm) else False

    support, resistance = find_support_resistance(high,low,close, min(60, n_candles - 5))
    risk = round(price-support,2) if support else 0
    reward = round(resistance-price,2) if resistance else 0
    rr_ratio = round(reward/risk,2) if risk>0 else 0

    # MACD zero-cross detection: MACD just turned positive from ≤0
    cm_prev_val = ml.iloc[-2] if len(ml) >= 2 else 0
    macd_zero_cross_up = bool(ok(cm) and cm > 0 and cm < 0.15 and ok(cm_prev_val) and cm_prev_val <= 0)

    # Mini MACD curve (last 20 values, normalised for sparkline)
    macd_curve = []
    n_curve = min(20, len(ml))
    if n_curve > 2:
        raw = [float(ml.iloc[-n_curve + j]) for j in range(n_curve) if ok(ml.iloc[-n_curve + j])]
        if raw:
            mn, mx = min(raw), max(raw)
            rng = mx - mn if mx != mn else 1
            macd_curve = [round((v - mn) / rng, 3) for v in raw]

    # ══════════════════════════════════════════════════════════
    # BUY SCORE — Revised Weights (max ~110 with all bonuses)
    # ══════════════════════════════════════════════════════════
    buy_score = 0; buy_breakdown = {}

    # 1. SMA(200) — 25 pts
    sma_p = bool(price > s200) if ok(s200) else False
    sma_pts = 25 if sma_p else 0; buy_score += sma_pts
    sma_desc = "Close > SMA(200)" if ok(s200) else f"N/A ({n_candles} candles < 200)"
    buy_breakdown["sma200"] = {"pass": sma_p, "pts": sma_pts, "max": 25,
                                "val": sf(s200), "desc": sma_desc}

    # 2. MACD INFLECTION — 30 pts (★ highest weight)
    #    Best case: MACD just crossed zero up (0 < MACD < 0.15) AND d²/dt² > 0
    #    Good case: slope just flipped positive
    #    Early case: decline decelerating (d²/dt² > 0 while d/dt < 0)
    macd_inf_pts = 0
    macd_inf_desc = ""
    if macd_zero_cross_up and macd_accel > 0:
        macd_inf_pts = 30
        macd_inf_desc = "MACD crossed 0↑ + accel ↑ (PRIME ENTRY)"
    elif slope_cross_up:
        macd_inf_pts = 20
        macd_inf_desc = "Slope flipped positive ↑"
    elif early_buy:
        macd_inf_pts = 10
        macd_inf_desc = "Decline slowing (d²>0, d<0)"
    macd_inf_pass = macd_inf_pts > 0
    buy_score += macd_inf_pts
    buy_breakdown["macd_inflection"] = {"pass": macd_inf_pass, "pts": macd_inf_pts, "max": 30,
                                         "val": sf(macd_slope, 4), "desc": macd_inf_desc or "No inflection detected"}

    # 3. RSI(14) — 15 pts + 5 bonus
    rsi_p = bool(30 < r14 <= 45) if ok(r14) else False
    rsi_deep = bool(r14 <= 35) if ok(r14) else False
    rsi_pts = (15 + (5 if rsi_deep else 0)) if rsi_p else 0; buy_score += rsi_pts
    buy_breakdown["rsi"] = {"pass": rsi_p, "pts": rsi_pts, "max": 20,
                             "val": sf(r14), "desc": "RSI 30-45" + (" +bonus ≤35" if rsi_deep else "")}

    # 4. Bollinger Bands — 10 pts + 5 bonus
    bb_p = bool(abm or tlb) if ok(cbm) else False
    bb_pts = (10 + (5 if tlb else 0)) if bb_p else 0; buy_score += bb_pts
    buy_breakdown["bollinger"] = {"pass": bb_p, "pts": bb_pts, "max": 15,
                                   "val": sf(cbm), "desc": "At/below mid BB" + (" +touched lower" if tlb else "")}

    # 5. ADX(14) — 10 pts + 5 bonus
    adx_p = bool(curr_adx > 25); adx_str = bool(curr_adx > 30)
    adx_pts = (10 + (5 if adx_str else 0)) if adx_p else 0; buy_score += adx_pts
    buy_breakdown["adx"] = {"pass": adx_p, "pts": adx_pts, "max": 15,
                             "val": sf(curr_adx), "desc": "ADX > 25" + (" +bonus >30" if adx_str else "")}

    # 6. OBV — 5 pts
    obv_p = bool(co > co5 and co > co20)
    obv_pts = 5 if obv_p else 0; buy_score += obv_pts
    buy_breakdown["obv"] = {"pass": obv_p, "pts": obv_pts, "max": 5,
                             "val": sf(co, 0), "desc": "OBV rising vs 5d & 20d"}

    buy_signal = "STRONG BUY" if buy_score >= 75 else ("MODERATE BUY" if buy_score >= 60 else "NO SIGNAL")

    sell_c = {
        "trend_break": bool(price<s200) if ok(s200) else False,
        "momentum_reversal": bool(r14>=65) if ok(r14) else False,
        "volatility_extreme": bool(price>=cbu) if ok(cbu) else False,
        "momentum_fade": bool(cm<cms and cmh<0) if ok(cm) and ok(cms) else False,
        "volume_weakness": bool(co<co5),
        "macd_slope_flip": slope_cross_dn,
    }
    sell_count = sum(sell_c.values())

    return {
        "price":sf(price),"sma200":sf(s200),"rsi":sf(r14),
        "bb_upper":sf(cbu),"bb_mid":sf(cbm),"bb_lower":sf(cbl),
        "macd":sf(cm,4),"macd_signal":sf(cms,4),"macd_hist":sf(cmh,4),
        "macd_slope":round(macd_slope,4),"macd_accel":round(macd_accel,4),
        "macd_phase":macd_phase,"macd_curve":macd_curve,
        "obv":sf(co,0),"adx":sf(curr_adx),
        "support":support,"resistance":resistance,"risk":risk,"reward":reward,"rr_ratio":rr_ratio,
        "buy_score":buy_score,"buy_signal":buy_signal,"buy_breakdown":buy_breakdown,
        "sell_conditions":sell_c,"sell_count":f"{sell_count}/6","sell_pct":sell_count/6*100,
        "is_buy":buy_score>=75,"is_moderate_buy":buy_score>=60,"is_sell":sell_count>=3,
    }

# ═══════════════════════════════════════════════════════════════
# SCAN — ALL NSE EQUITIES
# ═══════════════════════════════════════════════════════════════

def run_full_scan():
    global scan_results, is_scanning, scan_progress
    if not credentials_ok:
        logger.error("Cannot scan — not logged in")
        return
    if is_scanning: return
    is_scanning = True
    scan_results["status"] = "scanning"

    total = len(scan_instrument_list)
    scan_progress = {"current": 0, "total": total, "ok": 0, "errors": 0, "rate_limited": False}
    logger.info(f"Starting scan ({scan_mode.upper()}): {total} stocks...")
    t0 = time.time()
    buys, sells, all_s, errs = [], [], [], []

    portfolio = fetch_portfolio()
    ptokens = {h["token"] for h in portfolio}
    idx_counts = {n: 0 for n in INDEX_NAMES}
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 15  # abort if 15 in a row (likely rate limited)

    for i, inst in enumerate(scan_instrument_list):
        sym, token = inst["symbol"], inst["token"]
        name = inst.get("name", sym.replace("-EQ",""))
        clean = sym.replace("-EQ","")

        # Update live progress
        scan_progress["current"] = i + 1

        # Abort if too many consecutive errors (rate limit or session dead)
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            logger.error(f"Aborting scan: {consecutive_errors} consecutive errors. Rate limited?")
            scan_progress["rate_limited"] = True
            break

        try:
            df = fetch_candle_data(token)
            if df is None:
                # API returned nothing — this IS a potential rate limit issue
                consecutive_errors += 1
                errs.append({"symbol": sym, "error": "No data returned"})
                scan_progress["errors"] = len(errs)
                continue
            if len(df) < 50:
                # Too few candles — not an API error, just a thinly traded stock
                errs.append({"symbol": sym, "error": f"Only {len(df)} candles"})
                scan_progress["errors"] = len(errs)
                # Do NOT increment consecutive_errors — this is normal for small caps
                continue
            a = analyze_stock(df)
            if a is None:
                # Analysis failed but API worked — not a rate limit
                errs.append({"symbol": sym, "error": "Analysis returned None"})
                scan_progress["errors"] = len(errs)
                continue

            # Success — reset consecutive error counter
            consecutive_errors = 0

            indices = get_indices_for(clean, INDEX_MAP)
            stock = {"symbol": sym, "name": name, "token": token,
                     "in_portfolio": token in ptokens, "indices": indices, **a}
            all_s.append(stock)
            scan_progress["ok"] = len(all_s)

            for ix in indices:
                if ix in idx_counts: idx_counts[ix] += 1

            if a["is_buy"] or a["is_moderate_buy"]: buys.append(stock)
            if a["is_sell"] and token in ptokens: sells.append(stock)

            # Update scan_results live so frontend can show partial results
            if len(all_s) % 10 == 0:
                scan_results.update({
                    "total_scanned": len(all_s),
                    "buy_signals": sorted(buys, key=lambda x: x["buy_score"], reverse=True),
                    "sell_signals": sorted(sells, key=lambda x: x["sell_pct"], reverse=True),
                    "all_stocks": sorted(all_s, key=lambda x: x["buy_score"], reverse=True),
                    "index_counts": idx_counts.copy(),
                })

            time.sleep(0.5)  # 2 req/s — safe margin under 3/s limit
        except Exception as e:
            errs.append({"symbol": sym, "error": str(e)})
            scan_progress["errors"] = len(errs)
            consecutive_errors += 1

        if (i+1) % 50 == 0:
            elapsed_so_far = time.time() - t0
            rate = (i+1) / elapsed_so_far if elapsed_so_far > 0 else 0
            remaining = (total - i - 1) / rate / 60 if rate > 0 else 0
            logger.info(f"  {i+1}/{total} ({len(all_s)} ok, {len(errs)} err) ~{remaining:.0f}m left")

    elapsed = time.time() - t0
    scan_results = {
        "last_scan": datetime.now().isoformat(), "status": "complete",
        "scan_duration_sec": round(elapsed,1), "total_scanned": len(all_s),
        "buy_signals": sorted(buys, key=lambda x: x["buy_score"], reverse=True),
        "sell_signals": sorted(sells, key=lambda x: x["sell_pct"], reverse=True),
        "all_stocks": sorted(all_s, key=lambda x: x["buy_score"], reverse=True),
        "portfolio_holdings": portfolio, "errors": errs, "index_counts": idx_counts,
    }
    is_scanning = False
    logger.info(f"FULL SCAN DONE in {elapsed/60:.1f}min — {len(all_s)} stocks, {len(buys)} buys, {len(sells)} sells, {len(errs)} errors")
    for ix, cnt in idx_counts.items():
        if cnt > 0: logger.info(f"  {ix}: {cnt}")

# ═══════════════════════════════════════════════════════════════
# SCHEDULER
# ═══════════════════════════════════════════════════════════════

def scheduler():
    while True:
        now = datetime.now()
        wd = now.weekday() < 5
        ao = (now.hour==9 and now.minute>=15) or now.hour>=10
        bc = now.hour<15 or (now.hour==15 and now.minute<=30)
        if wd and ao and bc: run_full_scan()
        else: logger.info("Outside market hours")
        time.sleep(SCAN_INTERVAL)

# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route("/api/status")
def api_status():
    return jsonify({"connected": credentials_ok,
                    "instruments_total": len(instrument_list),
                    "instruments_scan": len(scan_instrument_list),
                    "scan_mode": scan_mode,
                    "last_scan": scan_results.get("last_scan"),
                    "scan_status": scan_results.get("status"), "is_scanning": is_scanning,
                    "index_counts": scan_results.get("index_counts", {}),
                    "missing": check_credentials()})

@app.route("/api/scan_mode", methods=["POST"])
def api_set_scan_mode():
    global scan_mode
    mode = request.json.get("mode", "top500").lower()
    if mode not in ("top500", "full"):
        return jsonify({"error": "Invalid mode. Use 'top500' or 'full'"}), 400
    scan_mode = mode
    apply_scan_mode()
    return jsonify({"scan_mode": scan_mode, "instruments_scan": len(scan_instrument_list)})

@app.route("/api/reconnect", methods=["POST"])
def api_reconnect():
    """Try to re-login and re-download instruments. Called from frontend."""
    global credentials_ok
    ok = create_session()
    if ok:
        if len(instrument_list) == 0:
            fetch_instrument_list()
        return jsonify({"status": "connected", "instruments_scan": len(scan_instrument_list)})
    return jsonify({"error": "Login failed. Check credentials or try again in a minute."}), 500

@app.route("/api/scan", methods=["POST"])
def api_trigger_scan():
    global credentials_ok
    # If not connected, try to reconnect first
    if not credentials_ok:
        if not create_session():
            return jsonify({"error": "Not connected to Angel One. Click Reconnect."}), 401
    # If instruments not loaded yet (startup failed), load them now
    if len(scan_instrument_list) == 0:
        fetch_instrument_list()
    if len(scan_instrument_list) == 0:
        return jsonify({"error": "No instruments loaded. Try reconnecting."}), 500
    if is_scanning:
        return jsonify({"status": "already_scanning"}), 409
    threading.Thread(target=run_full_scan, daemon=True).start()
    return jsonify({"status": "scan_started", "instruments": len(scan_instrument_list)})

@app.route("/api/results")
def api_results():
    return jsonify(scan_results)

@app.route("/api/progress")
def api_progress():
    return jsonify(scan_progress)

@app.route("/api/stock/<symbol>")
def api_stock_detail(symbol):
    for s in scan_results.get("all_stocks", []):
        if s["symbol"] == symbol or s["name"] == symbol: return jsonify(s)
    return jsonify({"error": "Not found"}), 404

@app.route("/")
def serve_frontend():
    return send_from_directory(".", "index.html")

# ── Watchlists ──
watchlists = {}

@app.route("/api/watchlists")
def api_get_watchlists():
    return jsonify(watchlists)

@app.route("/api/watchlists/<n>", methods=["PUT"])
def api_create_watchlist(n):
    if n not in watchlists: watchlists[n] = []
    return jsonify({"status": "created", "name": n})

@app.route("/api/watchlists/<n>", methods=["DELETE"])
def api_delete_watchlist(n):
    watchlists.pop(n, None)
    return jsonify({"status": "deleted"})

@app.route("/api/watchlists/<n>/add", methods=["POST"])
def api_watchlist_add(n):
    symbol = request.json.get("symbol", "")
    if n not in watchlists: watchlists[n] = []
    if symbol and symbol not in watchlists[n]: watchlists[n].append(symbol)
    return jsonify({"status": "added", "watchlist": watchlists[n]})

@app.route("/api/watchlists/<n>/remove", methods=["POST"])
def api_watchlist_remove(n):
    symbol = request.json.get("symbol", "")
    if n in watchlists and symbol in watchlists[n]: watchlists[n].remove(symbol)
    return jsonify({"status": "removed", "watchlist": watchlists.get(n, [])})

# ═══════════════════════════════════════════════════════════════
# INIT
# ═══════════════════════════════════════════════════════════════

def initialize():
    global credentials_ok
    missing = check_credentials()
    if missing:
        logger.error(f"Missing env vars: {', '.join(missing)}")
        credentials_ok = False
        return False

    # Try to login — if it fails, server still starts (user can reconnect from browser)
    if create_session():
        fetch_instrument_list()
    else:
        logger.warning("Initial login failed — server starting anyway. Use Reconnect button in browser.")

    threading.Thread(target=scheduler, daemon=True).start()
    logger.info(f"Ready. Scan mode: {scan_mode.upper()} ({len(scan_instrument_list)} stocks).")
    return True
