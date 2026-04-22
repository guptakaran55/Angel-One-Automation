# Graph Report - .  (2026-04-22)

## Corpus Check
- Corpus is ~16,910 words - fits in a single context window. You may not need a graph.

## Summary
- 141 nodes · 208 edges · 18 communities detected
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 22 edges (avg confidence: 0.83)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Cluster 0|Cluster 0]]
- [[_COMMUNITY_Cluster 1|Cluster 1]]
- [[_COMMUNITY_Cluster 2|Cluster 2]]
- [[_COMMUNITY_Cluster 3|Cluster 3]]
- [[_COMMUNITY_Cluster 4|Cluster 4]]
- [[_COMMUNITY_Cluster 5|Cluster 5]]
- [[_COMMUNITY_Cluster 6|Cluster 6]]
- [[_COMMUNITY_Cluster 7|Cluster 7]]
- [[_COMMUNITY_Cluster 8|Cluster 8]]
- [[_COMMUNITY_Cluster 9|Cluster 9]]
- [[_COMMUNITY_Cluster 10|Cluster 10]]
- [[_COMMUNITY_Cluster 11|Cluster 11]]
- [[_COMMUNITY_Cluster 12|Cluster 12]]
- [[_COMMUNITY_Cluster 13|Cluster 13]]
- [[_COMMUNITY_Cluster 14|Cluster 14]]
- [[_COMMUNITY_Cluster 15|Cluster 15]]
- [[_COMMUNITY_Cluster 16|Cluster 16]]
- [[_COMMUNITY_Cluster 17|Cluster 17]]

## God Nodes (most connected - your core abstractions)
1. `SignalScope` - 21 edges
2. `analyze_stock()` - 12 edges
3. `analyze_stock()` - 9 edges
4. `Composite Sell Score` - 9 edges
5. `Composite Buy Score` - 8 edges
6. `load_instrument_list()` - 7 edges
7. `create_session()` - 7 edges
8. `fetch_instrument_list()` - 7 edges
9. `run_full_scan()` - 7 edges
10. `run_nifty_scan()` - 5 edges

## Surprising Connections (you probably didn't know these)
- `SignalScope` --references--> `Pandas 2.2.3`  [INFERRED]
  INDICATORS.md → requirements.txt
- `SignalScope` --references--> `NumPy 2.2.3`  [INFERRED]
  INDICATORS.md → requirements.txt
- `SignalScope` --references--> `Requests 2.32.3`  [INFERRED]
  INDICATORS.md → requirements.txt
- `load_instrument_list()` --calls--> `get_scan_universe()`  [INFERRED]
  app.py → index_data.py
- `api_status()` --calls--> `get_nasdaq100_symbols()`  [INFERRED]
  app.py → us_market.py

## Hyperedges (group relationships)
- **Buy Score Six Indicators Combined** — indicators_buy_score, indicators_sma_trend, indicators_macd_inflection, indicators_rsi, indicators_bollinger_bands, indicators_adx, indicators_obv [EXTRACTED 1.00]
- **Golden Buy Three Simultaneous Conditions** — indicators_golden_buy, indicators_macd_inflection, indicators_sma_trend, indicators_macd_phases [EXTRACTED 1.00]
- **Position Sizing ATR Plus Support Resistance** — indicators_position_sizing, indicators_atr, indicators_support_resistance [EXTRACTED 1.00]
- **SignalScope Python Backend Stack** — indicators_signalscope, req_flask, req_pandas, req_numpy, req_requests, req_logzero, req_python_dotenv, req_yfinance [INFERRED 0.85]

## Communities

### Community 0 - "Cluster 0"
Cohesion: 0.14
Nodes (15): ATR 14 Average True Range, MIN_AVG_VOLUME Filter, NIFTY 500 Universe, Position Sizing 10K Risk Model, Price Dynamics Metrics, SignalScope, Support and Resistance Detection, Watchlist localStorage (+7 more)

### Community 1 - "Cluster 1"
Cohesion: 0.15
Nodes (13): api_index_chart(), api_sectors(), api_status(), Fetch index chart data with caching to avoid Yahoo rate limits during scans., Fetch all sector indices in one call. Returns list of chart data objects., run_nasdaq_scan(), fetch_index_chart_data(), fetch_us_candle_data() (+5 more)

### Community 2 - "Cluster 2"
Cohesion: 0.19
Nodes (13): ADX 14 Plus Directional Index, Bollinger Bands 20 2, Composite Buy Score, Dashboard Views Tabs, OBV On-Balance Volume, Rationale ADX must include directional filter, Rationale OBV sell weight doubled vs buy, Rationale RSI peak score at 35 (+5 more)

### Community 3 - "Cluster 3"
Cohesion: 0.18
Nodes (10): fetch_candle_data(), fetch_portfolio(), Fetch 2 years of daily OHLCV data from Yahoo Finance v8 chart API.     Uses dir, Fetch holdings from Zerodha. Returns list of dicts with tradingsymbol., run_nifty_scan(), download_nifty500(), get_scan_universe(), get_tags_for() (+2 more)

### Community 4 - "Cluster 4"
Cohesion: 0.18
Nodes (1): SignalScope v4.1 — Multi-Market Edition Scans NIFTY 500 (Yahoo Finance) + NASDA

### Community 5 - "Cluster 5"
Cohesion: 0.2
Nodes (11): analyze_stock(), calc_adx(), calc_bb(), calc_macd(), calc_obv(), calc_rsi(), calc_sma(), find_support_resistance() (+3 more)

### Community 6 - "Cluster 6"
Cohesion: 0.22
Nodes (10): analyze_stock(), calc_adx(), calc_bb(), calc_ema(), calc_macd(), calc_obv(), calc_rsi(), calc_sma() (+2 more)

### Community 7 - "Cluster 7"
Cohesion: 0.32
Nodes (8): api_reconnect(), api_reset(), api_trigger_scan(), create_session(), fetch_instrument_list(), Download NSE instruments from Angel One, filter to NIFTY 500 universe only., Try to re-login and re-download instruments. Called from frontend., Clear all scan results and reset to fresh state. Optionally re-login.

### Community 8 - "Cluster 8"
Cohesion: 0.25
Nodes (3): api_stop_scan(), SignalScope v3.1 — Scans ALL NSE equities (~500 stocks) Daily candles, 6 indica, Abort a running scan.

### Community 9 - "Cluster 9"
Cohesion: 0.33
Nodes (7): ensure_session(), fetch_candle_data(), fetch_portfolio(), Re-login if needed, with exponential backoff to avoid rate limit spiral., Fetch 730 days of daily candles for proper EMA warmup. Returns DataFrame, None (, run_full_scan(), scheduler()

### Community 10 - "Cluster 10"
Cohesion: 0.29
Nodes (7): Golden Buy Signal, MACD Inflection Indicator, MACD Momentum Phases, Rationale Golden Buy as coiled spring setup, Rationale SMA200 highest weight for trend, SMA Trend Indicator, NumPy 2.2.3

### Community 11 - "Cluster 11"
Cohesion: 0.4
Nodes (5): api_reconnect(), api_reset(), api_trigger_scan(), load_instrument_list(), Build NIFTY 500 scan list from index_data. Each stock mapped to a Yahoo Finance

### Community 12 - "Cluster 12"
Cohesion: 0.4
Nodes (5): initialize(), _verify_zerodha_token(), _zerodha_headers(), zerodha_holdings(), zerodha_set_token()

### Community 13 - "Cluster 13"
Cohesion: 0.4
Nodes (4): main(), api_status(), check_credentials(), initialize()

### Community 14 - "Cluster 14"
Cohesion: 0.67
Nodes (3): Launch scans for requested markets. Each market runs independently., run_full_scan(), scheduler()

### Community 15 - "Cluster 15"
Cohesion: 0.67
Nodes (3): Angel One SmartAPI, Angel One SmartAPI README, Requests 2.32.3

### Community 16 - "Cluster 16"
Cohesion: 1.0
Nodes (2): api_stop_scan(), Abort all running scans, or a specific market via ?market=nifty500

### Community 17 - "Cluster 17"
Cohesion: 1.0
Nodes (1): SignalScope v4.1 — Multi-Market Edition Run this file to start.  Usage:

## Knowledge Gaps
- **44 isolated node(s):** `SignalScope v4.1 — Multi-Market Edition Scans NIFTY 500 (Yahoo Finance) + NASDA`, `Build NIFTY 500 scan list from index_data. Each stock mapped to a Yahoo Finance`, `Fetch 2 years of daily OHLCV data from Yahoo Finance v8 chart API.     Uses dir`, `Fetch holdings from Zerodha. Returns list of dicts with tradingsymbol.`, `Analyze a stock DataFrame. Works for any market — just pass the right currency.` (+39 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Cluster 16`** (2 nodes): `api_stop_scan()`, `Abort all running scans, or a specific market via ?market=nifty500`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Cluster 17`** (2 nodes): `Start.py`, `SignalScope v4.1 — Multi-Market Edition Run this file to start.  Usage:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `get_scan_universe()` connect `Cluster 3` to `Cluster 11`, `Cluster 7`?**
  _High betweenness centrality (0.142) - this node is a cross-community bridge._
- **Why does `load_instrument_list()` connect `Cluster 11` to `Cluster 3`, `Cluster 4`, `Cluster 12`?**
  _High betweenness centrality (0.136) - this node is a cross-community bridge._
- **Why does `get_tags_for()` connect `Cluster 3` to `Cluster 9`?**
  _High betweenness centrality (0.127) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `SignalScope` (e.g. with `Flask 3.1.0` and `Flask-CORS 5.0.1`) actually correct?**
  _`SignalScope` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `SignalScope v4.1 — Multi-Market Edition Scans NIFTY 500 (Yahoo Finance) + NASDA`, `Build NIFTY 500 scan list from index_data. Each stock mapped to a Yahoo Finance`, `Fetch 2 years of daily OHLCV data from Yahoo Finance v8 chart API.     Uses dir` to the rest of the system?**
  _44 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Cluster 0` be split into smaller, more focused modules?**
  _Cohesion score 0.14 - nodes in this community are weakly interconnected._