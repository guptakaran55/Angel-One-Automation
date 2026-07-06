"""
scoring.py — SignalScope momentum / pullback / value scoring.

Ported from the SignalScope Android app (StockAnalyzer.kt, ValueAnalyzer.kt,
ScoringWeights.kt) as specified in SCORING_AI_PORT_SPEC.md.

Design note (why this file exists):
    The old web scanner rolled trend-continuation signals (SMA/ADX/OBV) and
    mean-reversion signals (oversold RSI, lower Bollinger band, dip to EMA21)
    into a SINGLE "buy score". Those two have conflicting motives — a strong
    momentum holding and a good pullback candidate pull the number in opposite
    directions. The Android app fixed this by computing two independent lenses:

        • PULLBACK  — "buy the dip inside an uptrend"
        • MOMENTUM  — trend-continuation + buy-and-hold quality

    plus a fundamental VALUE score. This module reproduces the EXACT thresholds,
    weights and bucket boundaries from the Android source — they are battle-tuned,
    do not "clean them up".

All scorers take a plain dict of pre-computed indicators (see app.analyze_stock
for how it is built) and return (score:int, signal:str, rows:list[dict]) where
each row is {"name","status","points","max","reason"} for a "Why this score?"
breakdown. Missing inputs score 0 for that bucket and never raise.
"""

import math


# ── helpers ──────────────────────────────────────────────────────────────────

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _rnd(x):
    """Half-up rounding to match the Android app's Math.round (see spec §9)."""
    return int(x + 0.5)


def _row(name, pts, mx, reason):
    pts = int(pts)
    status = "pass" if pts >= mx else ("fail" if pts <= 0 else "partial")
    return {"name": name, "status": status, "points": pts, "max": int(mx), "reason": reason}


def _num(ind, key, default=0.0):
    v = ind.get(key)
    if v is None:
        return default
    try:
        if isinstance(v, float) and math.isnan(v):
            return default
    except Exception:
        pass
    return v


# ── setup gate constants (spec §1.2 / §9) ─────────────────────────────────────

SETUP_MIN_ANGLE = -45.0
GOLDEN_BUY_MAX_ANGLE = 85.0
SETUP_SMA_MIN_SLOPE = 0.1
SETUP_MACD_PCTL_MAX = 75.0

PULLBACK_STRONG = 75
PULLBACK_MODERATE = 60
MOMENTUM_STRONG = 75
MOMENTUM_MODERATE = 60
CHERRY_STRONG = 75
CHERRY_MODERATE = 60


def _golden_buy(ind):
    """Qualified 'goldenBuy' setup gate (spec §1.2)."""
    macd_pctl = _num(ind, "macdPctl", 50.0)
    macd_accel = _num(ind, "macdAccel", 0.0)
    macd_angle = _num(ind, "macdSlopeAngle", 0.0)
    sma200_slope = _num(ind, "sma200Slope", 0.0)
    return bool(
        macd_pctl <= SETUP_MACD_PCTL_MAX
        and macd_accel > 0
        and macd_angle >= SETUP_MIN_ANGLE
        and macd_angle <= GOLDEN_BUY_MAX_ANGLE
        and sma200_slope >= SETUP_SMA_MIN_SLOPE
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. PULLBACK SCORE  (spec §1) — "buy the dip inside an uptrend"
# ══════════════════════════════════════════════════════════════════════════════

def pullback_score(ind):
    rows = []
    score = 0
    price = _num(ind, "price", 0.0)

    # 1.1 SMA trend
    sma200 = ind.get("sma200Val")
    sma_pass = sma200 is not None and price > sma200
    sma_pts = 25 if sma_pass else 0
    score += sma_pts
    rows.append(_row("SMA(200) trend", sma_pts, 25,
                     "Price > SMA200 (uptrend intact)" if sma_pass
                     else ("Price ≤ SMA200 (no uptrend)" if sma200 is not None else "SMA200 unavailable")))

    # 1.2 MACD inflection (+ golden / percentile bonuses)
    macd_pctl = _num(ind, "macdPctl", 50.0)
    macd_accel = _num(ind, "macdAccel", 0.0)
    golden = _golden_buy(ind)
    macd_zero_up = bool(ind.get("macdZeroCrossUp"))
    slope_up = bool(ind.get("slopeCrossUp"))
    early_buy = bool(ind.get("earlyBuy"))
    if golden:
        macd_inf = 30; inf_reason = "★ Golden buy setup (MACD hooking up in uptrend)"
    elif macd_zero_up and macd_accel > 0:
        macd_inf = 30; inf_reason = "MACD crossed 0↑ + accelerating (prime entry)"
    elif slope_up:
        macd_inf = 20; inf_reason = "MACD slope flipped positive ↑"
    elif early_buy:
        macd_inf = 10; inf_reason = "Decline slowing — early reversal"
    else:
        macd_inf = 0; inf_reason = "No MACD inflection"
    golden_bonus = 10 if golden else 0
    pctl_bonus = 3 if (macd_pctl <= 25.0 and macd_inf > 0) else 0
    if pctl_bonus:
        inf_reason += f" +{macd_pctl:.0f}%ile bonus"
    macd_total = macd_inf + golden_bonus + pctl_bonus
    score += macd_total
    rows.append(_row("MACD inflection", macd_total, 43, inf_reason))

    # 1.3 RSI graduated + flip bonus
    rsi = ind.get("rsiVal")
    rsi_pts = 0
    if rsi is not None:
        r = float(rsi)
        if 25.0 <= r <= 55.0:
            if r <= 35.0:
                rsi_pts = int(5 + 15 * (r - 25) / 10)
            else:
                rsi_pts = int(20 * (55 - r) / 20)
        elif r < 25.0:
            rsi_pts = 3
        else:
            rsi_pts = 0
        rsi_pts = int(_clamp(rsi_pts, 0, 20))
        rsi_reason = f"RSI {r:.1f} (pullback zone, peak ~35)" if rsi_pts else f"RSI {r:.1f} (outside pullback zone)"
    else:
        rsi_reason = "RSI unavailable"
    rsi_flip = bool(ind.get("rsiBuyFlip"))
    rsi_today = _num(ind, "rsiToday", 0.0)
    rsi_flip_pts = 8 if (rsi_flip and 25.0 <= rsi_today <= 45.0) else 0
    if rsi_flip_pts:
        rsi_reason += " +flip bonus (turning up)"
    score += rsi_pts + rsi_flip_pts
    rows.append(_row("RSI", rsi_pts + rsi_flip_pts, 28, rsi_reason))

    # 1.4 Bollinger
    below_mid = bool(ind.get("belowMidBand"))
    touched_lower = bool(ind.get("touchedLowerBand"))
    bb_pass = below_mid or touched_lower
    bb_pts = (10 + (5 if touched_lower else 0)) if bb_pass else 0
    score += bb_pts
    rows.append(_row("Bollinger", bb_pts, 15,
                     ("At/below mid band" + (" +touched lower band" if touched_lower else "")) if bb_pass
                     else "Above mid band (not a dip)"))

    # 1.5 ADX directional
    adx = _num(ind, "currAdx", 0.0)
    bullish = bool(ind.get("bullishTrend"))
    adx_strong = adx > 25.0
    adx_very = adx > 30.0
    if adx_strong and bullish:
        adx_pts = 15 + (5 if adx_very else 0)
        adx_reason = f"ADX {adx:.1f} · +DI>-DI (strong uptrend)" + (" +bonus>30" if adx_very else "")
    else:
        adx_pts = 0
        adx_reason = f"ADX {adx:.1f} — weak/wrong direction"
    score += adx_pts
    rows.append(_row("ADX", adx_pts, 20, adx_reason))

    # 1.6 OBV
    obv_pass = _obv_pass(ind)
    obv_pts = 5 if obv_pass else 0
    score += obv_pts
    rows.append(_row("OBV", obv_pts, 5, "OBV above 5d & 20d MA (accumulation)" if obv_pass else "OBV not confirming"))

    # 1.7 EMA(21) proximity
    ema21 = ind.get("ema21Val")
    ema_pts = 0
    if ema21 is not None:
        d = _num(ind, "ema21PctDiff", 0.0)
        if d < -3:
            ema_pts = 3; ema_reason = f"{d:.1f}% below EMA21 — deep dip, caution"
        elif -3.0 <= d <= 0.0:
            ema_pts = 10; ema_reason = f"{d:.1f}% below EMA21 — ideal pullback entry"
        elif 0.0 < d <= 2.0:
            ema_pts = int(10 * 0.8); ema_reason = f"{d:.1f}% above EMA21 — near ideal"
        elif 2.0 < d <= 4.0:
            ema_pts = int(10 * 0.5); ema_reason = f"{d:.1f}% above EMA21 — slightly stretched"
        elif 4.0 < d <= 7.0:
            ema_pts = int(10 * 0.2); ema_reason = f"{d:.1f}% above EMA21 — stretched"
        else:
            ema_pts = 0; ema_reason = f"{d:.1f}% above EMA21 — overextended"
    else:
        ema_reason = "EMA21 unavailable"
    score += ema_pts
    rows.append(_row("EMA(21) proximity", ema_pts, 10, ema_reason))

    if score >= PULLBACK_STRONG:
        signal = "STRONG PULLBACK"
    elif score >= PULLBACK_MODERATE:
        signal = "PULLBACK READY"
    else:
        signal = "NO SIGNAL"
    return int(score), signal, rows


# ══════════════════════════════════════════════════════════════════════════════
# 2. MOMENTUM SCORE  (spec §2) — trend continuation + buy-and-hold quality
# ══════════════════════════════════════════════════════════════════════════════

def _pct_slope(value, slope):
    return (slope / value * 100.0) if (value and value > 0) else 0.0


def momentum_score(ind):
    rows = []
    score = 0
    price = _num(ind, "price", 0.0)

    sma200 = ind.get("sma200Val")
    sma200_slope = _num(ind, "sma200Slope", 0.0)
    ema21 = ind.get("ema21Val")
    ema50 = ind.get("ema50Val")
    ema200 = ind.get("ema200Val")

    sma_slope_pct = _pct_slope(sma200, sma200_slope) if sma200 else 0.0
    ema21_slope_pct = _pct_slope(ema21, _num(ind, "ema21Slope", 0.0)) if ema21 else 0.0
    ema50_slope_pct = _pct_slope(ema50, _num(ind, "ema50Slope", 0.0)) if ema50 else 0.0
    ema200_slope_pct = _pct_slope(ema200, _num(ind, "ema200Slope", 0.0)) if ema200 else 0.0

    # 2.1 Trend stack (max 10)
    sma_pass = sma200 is not None and price > sma200
    sma_trend_ok = sma_pass or (sma200 is not None and price >= sma200 * 0.99 and sma200_slope > 0)
    raw_stack = ((30 if sma_trend_ok else 0)
                 + (25 if (sma_slope_pct > 0 or ema200_slope_pct > 0) else 0)
                 + (25 if ema50_slope_pct > 0 else 0)
                 + (20 if ema21_slope_pct > 0 else 0))
    trend_pts = _rnd(raw_stack / 100.0 * 10)
    score += trend_pts
    rows.append(_row("Trend stack", trend_pts, 10, f"{raw_stack}/100 of price>SMA200 + EMA slopes up"))

    # 2.2 MACD continuation (max 8)
    macd_zero_up = bool(ind.get("macdZeroCrossUp"))
    slope_up = bool(ind.get("slopeCrossUp"))
    macd_accel = _num(ind, "macdAccel", 0.0)
    macd_slope = _num(ind, "macdSlope", 0.0)
    macd_phase = ind.get("macdPhaseBull", "BEARISH")
    early_buy = bool(ind.get("earlyBuy"))
    if (macd_zero_up and macd_accel > 0) or slope_up:
        macd_pts = 8; macd_reason = "MACD crossing up / slope flipped"
    elif macd_phase == "BULLISH" and macd_slope > 0:
        macd_pts = _rnd(8 * 0.85); macd_reason = "Bullish MACD, rising"
    elif macd_slope > 0 and macd_accel >= 0:
        macd_pts = _rnd(8 * 0.70); macd_reason = "MACD rising, non-negative accel"
    elif early_buy and macd_accel > 0:
        macd_pts = _rnd(8 * 0.55); macd_reason = "Early reversal accelerating"
    else:
        macd_pts = 0; macd_reason = "No MACD continuation"
    score += macd_pts
    rows.append(_row("MACD continuation", macd_pts, 8, macd_reason))

    # 2.3 EMA21 slope (max 5)
    if ema21 is None:
        ema_slope_pts = 0; ema_slope_reason = "EMA21 unavailable"
    elif ema21_slope_pct > 0 and price >= ema21:
        ema_slope_pts = 5; ema_slope_reason = "EMA21 rising, price above it"
    elif ema21_slope_pct > 0:
        ema_slope_pts = _rnd(5 * 0.60); ema_slope_reason = "EMA21 rising"
    else:
        ema_slope_pts = 0; ema_slope_reason = "EMA21 flat/falling"
    score += ema_slope_pts
    rows.append(_row("EMA(21) slope", ema_slope_pts, 5, ema_slope_reason))

    # 2.4 Not overextended / room (max 3)
    d = _num(ind, "ema21PctDiff", 0.0)
    if 0.0 <= d <= 4.0:
        room_pts = 3
    elif -2.0 <= d < 0.0:
        room_pts = _rnd(3 * 0.80)
    elif 4.0 < d <= 7.0:
        room_pts = _rnd(3 * 0.45)
    elif -4.0 <= d < -2.0:
        room_pts = _rnd(3 * 0.35)
    else:
        room_pts = 0
    score += room_pts
    rows.append(_row("Room to run", room_pts, 3, f"{d:.1f}% from EMA21 (not overextended)" if room_pts else f"{d:.1f}% from EMA21 (overextended)"))

    # 2.5 RSI room (max 3)
    rsi = ind.get("rsiVal")
    if rsi is None:
        rsi_pts = 0; rsi_reason = "RSI unavailable"
    else:
        r = float(rsi)
        if 45.0 <= r <= 68.0:
            rsi_pts = 3
        elif 35.0 <= r < 45.0:
            rsi_pts = _rnd(3 * 0.60)
        elif 68.0 < r <= 75.0:
            rsi_pts = _rnd(3 * 0.45)
        elif r < 35.0:
            rsi_pts = _rnd(3 * 0.20)
        else:
            rsi_pts = 0
        rsi_reason = f"RSI {r:.1f} (healthy momentum band)" if rsi_pts else f"RSI {r:.1f} (extended)"
    score += rsi_pts
    rows.append(_row("RSI room", rsi_pts, 3, rsi_reason))

    # 2.6 OBV (max 4)
    obv_pass = _obv_pass(ind)
    obv_pts = 4 if obv_pass else 0
    score += obv_pts
    rows.append(_row("OBV", obv_pts, 4, "OBV above 5d & 20d MA" if obv_pass else "OBV not confirming"))

    # 2.7 Trend persistence (max 20)
    p = _num(ind, "trendPersistencePct", 0.0)
    if p >= 0.80:
        tp_pts = 20
    elif p >= 0.60:
        tp_pts = _rnd(20 * 0.75)
    elif p >= 0.40:
        tp_pts = _rnd(20 * 0.45)
    elif p >= 0.20:
        tp_pts = _rnd(20 * 0.20)
    else:
        tp_pts = 0
    score += tp_pts
    rows.append(_row("Trend persistence", tp_pts, 20, f"{p*100:.0f}% of last 200 bars above SMA200"))

    # 2.8 EMA cascade (max 15)
    cascade_ok = bool(ind.get("emaCascadeOk"))
    cascade_strong = bool(ind.get("emaCascadeStrong"))
    if cascade_strong:
        casc_pts = 15; casc_reason = "EMA21>50>200, all rising"
    elif cascade_ok:
        casc_pts = _rnd(15 * 0.60); casc_reason = "EMA21>50>200 stacked"
    else:
        casc_pts = 0; casc_reason = "EMAs not stacked"
    score += casc_pts
    rows.append(_row("EMA cascade", casc_pts, 15, casc_reason))

    # 2.9 Max drawdown (max 10)
    dd = _num(ind, "maxDrawdownPct", 1.0)
    if dd <= 0.10:
        dd_pts = 10
    elif dd <= 0.20:
        dd_pts = _rnd(10 * 0.60)
    elif dd <= 0.30:
        dd_pts = _rnd(10 * 0.30)
    else:
        dd_pts = 0
    score += dd_pts
    rows.append(_row("Max drawdown", dd_pts, 10, f"{dd*100:.0f}% max drawdown (lower is better)"))

    # 2.10 Long-term return (max 15)
    r6 = _num(ind, "return6mPct", 0.0)
    r12 = _num(ind, "return12mPct", 0.0)
    tranche6 = 6 if r6 >= 25 else (5 if r6 >= 15 else (3 if r6 >= 5 else 0))
    tranche12 = 9 if r12 >= 40 else (7 if r12 >= 25 else (5 if r12 >= 10 else 0))
    ret_pts = min(tranche6 + tranche12, 15)
    score += ret_pts
    rows.append(_row("Long-term return", ret_pts, 15, f"6m {r6:+.0f}% · 12m {r12:+.0f}%"))

    # 2.11 Sharpe-like (max 10)
    s = _num(ind, "sharpeLike", 0.0)
    if s >= 1.5:
        sh_pts = 10
    elif s >= 1.0:
        sh_pts = _rnd(10 * 0.70)
    elif s >= 0.5:
        sh_pts = _rnd(10 * 0.40)
    elif s > 0.0:
        sh_pts = _rnd(10 * 0.15)
    else:
        sh_pts = 0
    score += sh_pts
    rows.append(_row("Sharpe-like", sh_pts, 10, f"Risk-adj return ≈ {s:.2f}"))

    if score >= MOMENTUM_STRONG:
        signal = "STRONG MOMENTUM"
    elif score >= MOMENTUM_MODERATE:
        signal = "MOMENTUM READY"
    else:
        signal = "NO SIGNAL"
    return int(score), signal, rows


def _obv_pass(ind):
    cur = ind.get("obvCurrent")
    o5 = ind.get("obv5")
    o20 = ind.get("obv20")
    if cur is None or o5 is None or o20 is None:
        return False
    return bool(cur > o5 and cur > o20)


# ══════════════════════════════════════════════════════════════════════════════
# 3. COMPOSITE  (spec §3)
# ══════════════════════════════════════════════════════════════════════════════

def composite(pullback, momentum):
    cherry = max(pullback, momentum)
    if pullback >= 60 and momentum >= 60:
        entry = "PULLBACK + MOMENTUM"
    elif momentum >= 60:
        entry = "MOMENTUM"
    elif pullback >= 60:
        entry = "PULLBACK"
    else:
        entry = "NONE"

    if cherry >= CHERRY_STRONG and entry == "MOMENTUM":
        signal = "STRONG MOMENTUM"
    elif cherry >= CHERRY_STRONG and entry == "PULLBACK":
        signal = "STRONG PULLBACK"
    elif cherry >= CHERRY_STRONG:
        signal = "STRONG ENTRY"
    elif cherry >= CHERRY_MODERATE and entry == "MOMENTUM":
        signal = "MOMENTUM READY"
    elif cherry >= CHERRY_MODERATE and entry == "PULLBACK":
        signal = "PULLBACK READY"
    elif cherry >= CHERRY_MODERATE:
        signal = "ENTRY READY"
    else:
        signal = "NO SIGNAL"
    return {"cherryPoints": int(cherry), "entryMode": entry, "signal": signal}


# ══════════════════════════════════════════════════════════════════════════════
# 4. VALUE SCORE  (spec §4) — needs Yahoo fundamentals
# ══════════════════════════════════════════════════════════════════════════════

def value_score(f, price, sector_median_pe=None):
    """
    f: dict of fundamentals (see app.fetch_fundamentals), normalized to this
    contract:
      • margins (gross/operating/profit), growth (revenue/earnings), dividendYield
        → PERCENT (10.0 == 10%)
      • returnOnEquity → FRACTION (0.35 == 35%)
      • debtToEquity → RATIO (0.3 == 30%)
      • PE/forwardPE/priceToBook/enterpriseToEbitda/currentRatio → raw ratios
      • totalCash/totalDebt/operatingCashflow/freeCashflow/netIncome → currency
      • fiftyTwoWeekLow → price
    Missing fields are skipped (score 0 for that part).
    Returns (score:int 0-100, rating:str, rows:list[dict]).
    """
    rows = []
    price = price or 0.0

    def g(k):
        return f.get(k)

    # 4.1 Valuation (cap 30)
    valuation = 0
    pe = g("trailingPE")
    fwd_pe = g("forwardPE")
    pb = g("priceToBook")
    ev_ebitda = g("enterpriseToEbitda")
    if pe and pe > 0:
        valuation += 10 if pe < 12 else (7 if pe < 18 else (4 if pe < 25 else 0))
        if sector_median_pe and sector_median_pe > 0:
            valuation += 8 if pe <= sector_median_pe * 0.75 else (5 if pe < sector_median_pe else 0)
    if fwd_pe and pe and fwd_pe > 0 and pe > 0 and fwd_pe <= pe * 0.90:
        valuation += 4
    if pb and pb > 0:
        valuation += 6 if pb < 1.5 else (3 if pb < 2.0 else 0)
    if ev_ebitda and ev_ebitda > 0:
        valuation += 8 if ev_ebitda < 8 else (4 if ev_ebitda < 12 else 0)
    valuation = int(_clamp(valuation, 0, 30))
    rows.append(_row("Valuation", valuation, 30,
                     f"P/E {pe:.1f}" if (pe and pe > 0) else "P/E n/a"))

    # 4.2 Quality (cap 30)
    quality = 0
    de = g("debtToEquity")
    total_cash = g("totalCash")
    total_debt = g("totalDebt")
    current_ratio = g("currentRatio")
    roe = g("returnOnEquity")
    profit_margin = g("profitMargins")
    operating_margin = g("operatingMargins")
    cfo = g("operatingCashflow")
    ni = g("netIncome")
    fcf = g("freeCashflow")
    if de is not None:
        quality += 6 if de < 0.3 else (4 if de < 1.0 else (2 if de < 2.0 else 0))
    if total_cash is not None and total_debt is not None and (total_cash - total_debt) >= 0:
        quality += 4
    if current_ratio:
        quality += 3 if current_ratio >= 1.5 else (1 if current_ratio >= 1.0 else 0)
    roe_pct = (roe * 100) if roe is not None else None  # roe passed as a fraction (0.35 == 35%)
    if roe_pct:
        quality += 6 if roe_pct >= 15 else (3 if roe_pct >= 10 else 0)
    if profit_margin:
        quality += 4 if profit_margin >= 10 else (2 if profit_margin >= 5 else 0)
    if operating_margin:
        quality += 3 if operating_margin >= 15 else (1 if operating_margin >= 8 else 0)
    if cfo and ni and ni > 0 and cfo > ni:
        quality += 4
    if fcf and fcf > 0:
        quality += 3
    quality = int(_clamp(quality, 0, 30))
    rows.append(_row("Quality", quality, 30,
                     f"ROE {roe_pct:.0f}%" if roe_pct else "quality metrics"))

    # 4.3 Growth (cap 18)
    growth = 0
    rg = g("revenueGrowth")
    eg = g("earningsGrowth")
    gross_margin = g("grossMargins")
    if rg:
        growth += 7 if rg >= 12 else (4 if rg >= 4 else (2 if rg > 0 else 0))
    if eg:
        growth += 7 if eg >= 12 else (4 if eg >= 4 else (2 if eg > 0 else 0))
    if gross_margin and gross_margin >= 35:
        growth += 2
    if profit_margin and profit_margin >= 10 and eg and eg > 0:
        growth += 2
    growth = int(_clamp(growth, 0, 18))
    rows.append(_row("Growth", growth, 18,
                     f"rev {rg:+.0f}% · eps {eg:+.0f}%" if (rg is not None or eg is not None) else "growth n/a"))

    # 4.4 Yield (cap 10)
    yield_ = 0
    div_yield = g("dividendYield")
    if div_yield and div_yield > 0:
        yield_ += 6 if div_yield >= 3 else (3 if div_yield >= 1 else 0)
    yield_ = int(_clamp(yield_, 0, 10))
    rows.append(_row("Yield", yield_, 10, f"Div yield {div_yield:.1f}%" if div_yield else "No dividend"))

    # 4.5 Discount (cap 12)
    discount = 0
    low52 = g("fiftyTwoWeekLow")
    if low52 and low52 > 0 and price > 0:
        pct_from_low = (price - low52) / low52 * 100
        discount += 12 if pct_from_low < 15 else (5 if pct_from_low < 30 else 0)
        disc_reason = f"{pct_from_low:.0f}% above 52w low"
    else:
        disc_reason = "52w low n/a"
    discount = int(_clamp(discount, 0, 12))
    rows.append(_row("Discount", discount, 12, disc_reason))

    # 4.6 Penalties, then clamp
    score = valuation + quality + growth + yield_ + discount
    penalties = []
    if rg is not None and rg < -5:
        score -= 5; penalties.append("rev<-5% (-5)")
    if eg is not None and eg < -10:
        score -= 7; penalties.append("eps<-10% (-7)")
    if fcf is not None and fcf < 0:
        score -= 4; penalties.append("FCF<0 (-4)")
    score = int(_clamp(score, 0, 100))
    if penalties:
        rows.append(_row("Penalties", 0, 0, ", ".join(penalties)))

    # 4.7 Rating
    if score >= 80:
        rating = "DEEP VALUE"
    elif score >= 65:
        rating = "MODERATE VALUE"
    elif score >= 50:
        rating = "MILD VALUE"
    else:
        rating = "NOT ATTRACTIVE"
    return score, rating, rows
