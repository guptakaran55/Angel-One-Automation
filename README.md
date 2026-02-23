| Condition Type | Description |
|----------------|-------------|
| **BUY FLIP** | Slope just turned positive after 2+ days negative |
| **EARLY BUY** | Slope still negative but decelerating (d² > 0, d < 0) |
| **SELL FLIP** | Slope just turned negative after 2+ days positive |
| **EARLY SELL** | Slope still positive but decelerating |
| **BULLISH** | Slope > 0, acceleration > 0 |
| **BEARISH** | Slope < 0, acceleration < 0 |
| **NEUTRAL** | Mixed signals |



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
| **Sell** | Sell conditions triggered (X/6). Red badge |
| **MACD** | 20-day sparkline + phase label + slope (d/dt) + current MACD value |
| **Price** | Daily velocity (%), acceleration, up-day streak |
| **R:R** | Reward-to-risk ratio from support/resistance |
| **ATR Risk** | Stop loss distance measured in ATRs. Green ≥1.5, amber ≥1, red <1 |
| **Capital** | Capital required + potential profit (assumes ₹10K risk per trade) |
| **ROC%** | Return on capital — profit as % of capital deployed |
| **Signal** | Final label: ★ GOLDEN, STRONG BUY, MOD BUY, SELL, or HOLD |

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

### 5. ADX(14) — max 15 pts

**What:** Is there a strong directional trend (in either direction)?

| Condition | Points |
|-----------|--------|
| ADX > 25 | 10 |
| ADX > 30 | +5 bonus |
| ADX ≤ 25 | 0 |

**Why it matters:** ADX measures trend strength, not direction. A high ADX with a positive buy score means the stock is trending *and* the trend is in your favor. Low ADX means the stock is chopping sideways — signals from other indicators are less reliable in a rangebound market.

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

## Sell Conditions (6 binary checks)

Each condition is independently true or false. Shown as red/grey dots in the dashboard. Sell alert triggers when ≥3 of 6 are true **and** the stock is in your portfolio.

| # | Condition | Trigger |
|---|-----------|---------|
| 1 | **Trend Break** | Close < SMA(200) — lost the long-term uptrend |
| 2 | **Momentum Reversal** | RSI ≥ 65 — overbought territory |
| 3 | **Volatility Extreme** | Price ≥ upper Bollinger Band — stretched above 2σ |
| 4 | **Momentum Fade** | MACD < signal line AND histogram negative — bearish crossover |
| 5 | **Volume Weakness** | OBV lower than 5 days ago — volume leaving |
| 6 | **MACD Slope Flip** | MACD slope turned negative after 2+ days positive |

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
| **Sell** | Stocks with ≥ 3/6 sell conditions AND in your portfolio |
| **Portfolio** | Your Angel One holdings with current signals overlaid |

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