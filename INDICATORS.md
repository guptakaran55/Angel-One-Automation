# SignalScope — Indicator & Scoring Reference

SignalScope scans the NIFTY 500 universe daily using 6 technical indicators combined into a weighted buy score (max ~118 pts), 6 binary sell conditions, MACD momentum phase detection, and ATR-based position sizing. This document explains every metric shown in the dashboard.

---

## Scan Universe & Filters

**Stocks scanned:** NIFTY 500 constituents (downloaded live from NSE, with hardcoded fallback).

**Pre-analysis filters** — a stock is skipped entirely if:
- Fewer than 50 daily candles available
- 20-day average volume < 100,000 shares (configurable via `MIN_AVG_VOLUME`)

Stocks that pass filters are tagged with their index memberships (NIFTY 50, NIFTY 100, BSE 100) for quick identification.

---

## Main Table Columns

| Column | What it shows |
|--------|--------------|
| **Stock** | Name, symbol, volume, index tags, portfolio/watchlist status |
| **Price** | Last closing price |
| **Buy** | Composite buy score (0–118). Green badge. ≥75 = Strong Buy, ≥60 = Moderate Buy |
| **Sell** | Composite sell score (0–108). Red badge. ≥65 = Strong Sell, ≥45 = Moderate Sell |
| **MACD** | 20-day sparkline + phase label + slope (d/dt) + current MACD value |
| **Price** | Daily velocity (%), acceleration, up-day streak |
| **R:R** | Reward-to-risk ratio from support/resistance |
| **ATR Risk** | Stop loss distance measured in ATRs. Green ≥1.5, amber ≥1, red <1 |
| **Capital** | Capital required + potential profit (assumes ₹10K risk per trade) |
| **ROC%** | Return on capital — profit as % of capital deployed |
| **Signal** | Final label: ★ GOLDEN, STRONG BUY, MOD BUY, STRONG SELL, MOD SELL, or HOLD |

---

## Buy Score Breakdown (max ~118 pts)

The buy score is the sum of 6 indicator scores. Each indicator contributes points independently. Higher is more bullish.

### 1. SMA Trend — max 25 pts

**What:** Is price above its long-term moving average?

| Condition | Points |
|-----------|--------|
| Close > SMA(200) | 25 |
| Close > SMA(50) *(fallback for stocks with <200 candles)* | 15 |
| Neither available | 0 |

**Why it matters:** A stock trading above its 200-day average is in a confirmed uptrend. Buying below SMA(200) means buying against the primary trend. Newer listings use SMA(50) with reduced weight since the signal is less reliable with less data.

**Also tracked:** `d(SMA)/dt` — the daily slope of SMA(200). Positive = trend strengthening. Used in Golden Buy detection.

---

### 2. MACD Inflection — max 43 pts ★ (highest weight)

**What:** Where is the MACD line in its cycle, and is momentum shifting?

This is the most complex indicator in SignalScope. It uses the MACD line (12,26,9), its first derivative (slope), second derivative (acceleration), and its position relative to its own 1-year range.

**Scoring tiers:**

| Condition | Points | Description |
|-----------|--------|-------------|
| ★ Golden Buy | 30 + 10 bonus | MACD near 1Y low + slope flat + SMA(200) rising |
| Prime Entry | 30 | MACD just crossed zero upward + acceleration positive |
| Buy Flip | 20 | Slope flipped positive after 2+ days negative |
| Early Buy | 10 | Slope still negative but decelerating (d² > 0) |
| No inflection | 0 | — |

**Percentile bonus:** +3 pts if MACD is in the bottom 25% of its 1-year range (regardless of which tier above).

#### MACD Derivatives Explained

- **MACD value** — The raw MACD line value. Shown on the sparkline in the table.
- **d(MACD)/dt** (slope) — Today's MACD minus yesterday's. Positive = momentum increasing. This is the green/red number next to the sparkline.
- **d²(MACD)/dt²** (acceleration) — Change in slope. Positive acceleration while slope is negative = the decline is slowing down, which is the earliest detectable sign of a potential reversal.
- **MACD %ile (1Y)** — Where the current MACD sits in its 1-year range (0% = at yearly low, 100% = at yearly high). Low percentile during an uptrend = potential buying opportunity.
- **MACD % of 1Y low** — How close the current MACD is to its worst reading of the year, as a percentage. 60%+ triggers Golden Buy consideration.

#### The Sparkline

The mini chart in the MACD column shows the last 20 days of MACD values. The red dashed line is the zero line, positioned at its true relative location (not centered). The dot marks today's value.

---

### 3. RSI(14) — max 20 pts

**What:** Is the stock in a favorable momentum zone for buying?

SignalScope uses **graduated scoring** — no cliff effects. Points peak at RSI 35 and taper linearly in both directions:

| RSI Range | Points | Interpretation |
|-----------|--------|---------------|
| < 25 | 3 | Deeply oversold — potential falling knife, proceed with caution |
| 25 → 35 | 5 → 20 | Approaching optimal buy zone, points increase linearly |
| **35** | **20** | **Peak score — best statistical buy zone** |
| 35 → 55 | 20 → 0 | Leaving buy zone, points decrease linearly |
| > 55 | 0 | Above buy zone |

**Why 35 is the peak:** RSI around 30-40 typically represents stocks that have pulled back within an uptrend — not yet oversold enough to signal panic, but cheap enough to offer good entry. RSI below 25 often indicates something structurally wrong.

---

### 4. Bollinger Bands (20,2) — max 15 pts

**What:** Is price at or below statistical fair value, and has it recently tested the lower extreme?

| Condition | Points |
|-----------|--------|
| Price at or below middle band | 10 |
| Price touched lower band in last 5 days | +5 bonus |
| Price above middle band | 0 |

**Why it matters:** The middle Bollinger Band is a 20-day moving average. Price below it suggests the stock is trading below its recent average — a potential value zone. Touching the lower band (2 standard deviations below) often marks short-term exhaustion.

---

### 5. ADX(14) + Directional Index — max 20 pts

**What:** Is there a strong trend **and is it bullish?**

ADX alone only measures trend *strength*. A stock crashing in a strong downtrend also has high ADX. SignalScope pairs ADX with the Directional Indicators (+DI and -DI) to ensure points are only awarded for **confirmed uptrends**.

| Condition | Points | Meaning |
|-----------|--------|---------|
| ADX > 25 AND +DI > -DI | 15 | Strong uptrend confirmed |
| ADX > 30 AND +DI > -DI | +5 bonus | Very strong uptrend |
| ADX > 25 AND -DI > +DI | **0** | Strong **downtrend** — no points awarded |
| ADX ≤ 25 | 0 | Trend too weak to be meaningful |

**+DI (Plus Directional Indicator):** Measures upward price pressure.
**-DI (Minus Directional Indicator):** Measures downward price pressure.
When +DI > -DI, bulls are in control. When -DI > +DI, bears are in control.

**Why direction matters:** Without this check, a stock in freefall with ADX 35 would score 15 buy points — actively encouraging you to buy into a confirmed downtrend. The directional filter prevents this.

**Sell condition:** When ADX > 25 AND -DI > +DI, the "Strong Downtrend" sell condition is triggered (see Sell Conditions below).

---

### 6. OBV (On-Balance Volume) — max 5 pts

**What:** Is volume confirming the price move?

| Condition | Points |
|-----------|--------|
| OBV higher than both 5 days ago and 20 days ago | 5 |
| Otherwise | 0 |

**Why it matters:** OBV tracks cumulative volume flow. Rising OBV means more volume on up days than down days — institutions are accumulating. A price rise without OBV confirmation is suspect.

---

### Signal Thresholds

| Total Score | Signal |
|-------------|--------|
| ≥ 75 | **STRONG BUY** |
| ≥ 60 | **MODERATE BUY** |
| < 60 | **NO SIGNAL** |

---

## Sell Score Breakdown (max ~108 pts)

The sell score mirrors the buy score — a weighted composite of 6 indicator scores. Higher means more bearish.

**Structural gate:** STRONG SELL (≥65) requires at least one *structural* signal (SMA break or ADX bearish). Without structural confirmation, the max label is MODERATE SELL — even if momentum signals alone push above 65. This prevents false sell signals in intact uptrends.

### 1. SMA Trend Break — max 25 pts

| Condition | Points |
|-----------|--------|
| Close < SMA(200) | 25 |
| Close ≥ SMA(200) | 0 |

**Why highest weight:** Once price closes below the 200-day moving average, the uptrend is statistically broken. This is the most reliable daily trend-change signal.

### 2. MACD Inflection (Sell) — max 28 pts

Uses slope and acceleration to detect bearish momentum shifts *early* (not lagging crossover checks):

| Condition | Points |
|-----------|--------|
| MACD crossed below zero + deceleration (PRIME EXIT) | 25 |
| Slope flipped negative after 2+ positive days (SELL FLIP) | 18 |
| Rise slowing — d²<0 while d>0 (EARLY SELL) | 8 |
| +3 bonus if MACD is in top 25% of its 1-year range during inflection | +3 |

### 3. RSI Overheated — max 20 pts (graduated)

| RSI Range | Points | Description |
|-----------|--------|-------------|
| > 85 | 10 | Extremely overbought — parabolic risk, could snap either way |
| 70–85 | 20 | Overbought — max sell points |
| 65–69 | 15 | Approaching overbought |
| 60–64 | 8 | Elevated — early warning |
| < 60 | 0 | Below sell zone |

### 4. Bollinger Stretch — max 10 pts

| Condition | Points |
|-----------|--------|
| Price ≥ upper Bollinger Band | 10 |
| Touched upper band in last 5 days (but now below) | 5 |
| Below upper band | 0 |

### 5. ADX + Bearish Direction — max 15 pts

| Condition | Points |
|-----------|--------|
| ADX > 30 AND -DI > +DI | 15 |
| ADX > 25 AND -DI > +DI | 10 |
| ADX > 25 but +DI > -DI (uptrend) | 0 |
| ADX ≤ 25 (weak trend) | 0 |

### 6. OBV Declining — max 10 pts

| Condition | Points |
|-----------|--------|
| OBV below both 5-day and 20-day levels | 10 |
| Otherwise | 0 |

**Why doubled from buy side (5→10):** On the sell side, declining OBV confirms distribution — institutional selling shows in volume before price drops.

### Sell Signal Thresholds

| Total Score | Signal | Requirement |
|-------------|--------|-------------|
| ≥ 65 | **STRONG SELL** | Must have at least one structural signal (SMA break or ADX bearish) |
| ≥ 45 | **MODERATE SELL** | No structural requirement |
| < 45 | **NO SIGNAL** | — |

### Portfolio Verdicts (Zerodha tab only)

When viewing your Zerodha holdings, the sell score is combined with unrealized profit for portfolio-specific labels:

| Condition | Verdict |
|-----------|---------|
| Sell score ≥ 65 | **SELL NOW** |
| Sell score ≥ 45 | **MOD SELL** |
| Sell score ≥ 30 AND unrealized gain ≥ 25% | **BOOK PROFIT?** |
| Otherwise | **HOLD** |

---

## MACD Momentum Phases

Every stock is assigned one of 7 momentum phases based on MACD slope and acceleration:

| Phase | Slope (d/dt) | Acceleration (d²/dt²) | What it means |
|-------|-------------|----------------------|---------------|
| **BUY FLIP** | Just turned + | Was − for 2+ days | Momentum confirmed bullish — strongest buy signal |
| **EARLY BUY** | Still − | Turning + | Decline slowing — potential bottom forming |
| **SELL FLIP** | Just turned − | Was + for 2+ days | Momentum confirmed bearish — strongest sell signal |
| **EARLY SELL** | Still + | Turning − | Rise slowing — potential top forming |
| **BULLISH** | + | + | Rising momentum with positive acceleration |
| **BEARISH** | − | − | Falling momentum with negative acceleration |
| **NEUTRAL** | Mixed | Mixed | No clear directional signal |

**Noise filter:** A flip requires the prior slope to have been directional for at least 2 consecutive days with a minimum magnitude of 0.01. This prevents false flips on choppy, sideways stocks.

---

## ★ Golden Buy

The rarest and highest-conviction signal. Requires all three conditions simultaneously:

1. **MACD ≥ 60% of its 1-year low** — momentum has deeply corrected
2. **MACD slope ≤ 0.2** — momentum is flat (not falling further, not yet rising)
3. **SMA(200) slope > 0.1** — the long-term trend is still rising

**Interpretation:** The stock's long-term trend is intact (SMA rising), but short-term momentum has pulled back to near its worst levels of the year and stopped declining. This is the "coiled spring" setup — when the MACD eventually turns up from here, the move tends to be significant.

Scores 30 + 10 bonus = 40 pts from the MACD indicator alone.

---

## Price Dynamics

Shown in the Price column and the detail modal:

| Metric | Calculation | What it tells you |
|--------|------------|-------------------|
| **Daily Velocity** | (today - yesterday) / yesterday × 100 | How fast the price is moving right now (%) |
| **Acceleration** | Today's velocity - yesterday's velocity | Is the move speeding up or slowing down? |
| **3-day ROC** | (today - 3 days ago) / 3 days ago × 100 | Short-term directional bias |
| **Up Streak** | Consecutive up-close days (max 5 checked) | Shown as "↑3d" badge when ≥ 3 |

---

## Support & Resistance

Calculated using **swing point detection with clustering**:

1. Scan the last 60 trading days for swing highs and lows (using a 5-bar window on each side)
2. Cluster nearby levels within 1.5% of each other (so ₹99 and ₹100 merge into one level)
3. Nearest cluster below price = support; nearest above = resistance
4. Falls back to classic pivot points (R1/S1) if no swing points found

**Risk** = price − support (how far you'd fall to the nearest floor)  
**Reward** = resistance − price (how far you'd rise to the nearest ceiling)  
**R:R ratio** = reward ÷ risk

---

## Position Sizing (₹10K Risk Model)

All position sizing assumes a **fixed ₹10,000 maximum loss per trade**. You can mentally scale to your actual risk tolerance.

| Metric | Calculation | How to read it |
|--------|------------|----------------|
| **ATR(14)** | 14-day exponential average of daily high−low range | The stock's "normal" daily movement in rupees |
| **Risk in ATRs** | (price − support) ÷ ATR | How many normal daily moves to your stop loss |
| **Position Size** | ₹10,000 ÷ risk | Number of shares to buy |
| **Capital Needed** | position size × price | Total rupees deployed |
| **Potential Profit** | position size × reward | Rupees earned if resistance is hit |
| **ROC%** | potential profit ÷ capital needed × 100 | Return on capital — the efficiency metric |

### ATR Risk Quality Guide

| ATR Risk | Color | Meaning |
|----------|-------|---------|
| ≥ 1.5 | 🟢 Green | Stop is 1.5+ normal daily moves away — meaningful level, unlikely to be hit by noise |
| 1.0–1.5 | 🟡 Amber | Borderline — the stock could hit your stop on a normal volatile day |
| < 1.0 | 🔴 Red + ⚠ | Stop is less than one normal daily move away — very likely to be triggered by random noise. Consider widening your stop or skipping the trade |

---

## Dashboard Views

| Tab | What it shows |
|-----|--------------|
| **All** | Every analyzed stock, sorted by buy score |
| **🎯 Setups** | Golden Buy matches + stocks with SMA(200) pass and BUY FLIP or EARLY BUY phase |
| **Buy** | Stocks scoring ≥ 60 (Moderate + Strong buys) |
| **Sell** | Stocks scoring ≥ 45 sell points AND in your portfolio |
| **Portfolio** | Your Angel One holdings with current signals overlaid |
| **💰 Zerodha** | Zerodha holdings with sell-focused verdicts, pie chart, and profit protection |

### Setups Tab Filters

The Setups tab includes adjustable sliders for Golden Buy parameters:
- **MACD % of 1Y low ≥** — How deep the MACD must have dipped (default 60%)
- **d(MACD)/dt range** — Acceptable slope range (default -0.5 to 0.2)
- **d(SMA)/dt >** — Minimum SMA(200) slope (default 0.1)

Narrowing these filters gives fewer but higher-conviction setups.

---

## Watchlists

Watchlists are stored in your browser's localStorage (not on the server). They persist across sessions on the same browser but are not synced across devices.

- Create watchlists, add/remove stocks from the detail modal
- Click a watchlist chip to filter the table to just those stocks
- All technical data updates on each scan — watchlists just filter the view

---

## Scan Mechanics

- **Data source:** Angel One SmartAPI via `smartapi-python`
- **Candle history:** 730 days (2 years) for proper EMA warmup
- **Scan pacing:** Adaptive — starts at 0.35s per stock, backs off on rate limits, recovers on success
- **Session refresh:** Every 150 stocks to prevent token expiry
- **Auto-scheduling:** Scans automatically during market hours (9:15 AM – 3:30 PM IST, weekdays)
- **Volume filter:** Stocks with 20-day average volume < 100K are skipped (not analyzed)

---

## Quick Reference: What Makes a Good Setup?

The strongest buy signals typically combine:

✅ Price above SMA(200) — uptrend intact  
✅ RSI 30–40 — pulled back but not broken  
✅ MACD phase = BUY FLIP or EARLY BUY — momentum turning  
✅ MACD percentile < 25% — deeply corrected within the uptrend  
✅ ADX > 25 — strong trend, not choppy  
✅ R:R ≥ 2:1 — reward is at least double the risk  
✅ ATR Risk ≥ 1.5 — stop is at a meaningful level  
✅ ROC% ≥ 10% — capital efficiency is worthwhile  

A stock hitting all of these is rare — which is exactly the point. SignalScope scans 500 stocks so you don't have to.