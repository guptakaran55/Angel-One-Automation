---
type: community
cohesion: 0.15
members: 14
---

# Cluster 0

**Cohesion:** 0.15 - loosely connected
**Members:** 14 nodes

## Members
- [[Fetch all sector indices in one call. Returns list of chart data objects.]] - rationale - app.py
- [[Fetch daily OHLCV data from Yahoo Finance.     Returns DataFrame with same form]] - rationale - us_market.py
- [[Fetch index chart data using Yahoo Finance v8 chart API directly.     Bypasses]] - rationale - us_market.py
- [[Fetch index chart data with caching to avoid Yahoo rate limits during scans.]] - rationale - app.py
- [[Return list of NASDAQ 100 symbols.]] - rationale - us_market.py
- [[US Market Data Module — yfinance integration for SignalScope. Provides NASDAQ 1]] - rationale - us_market.py
- [[api_index_chart()]] - code - app.py
- [[api_sectors()]] - code - app.py
- [[api_status()]] - code - app.py
- [[fetch_index_chart_data()]] - code - us_market.py
- [[fetch_us_candle_data()]] - code - us_market.py
- [[get_nasdaq100_symbols()]] - code - us_market.py
- [[run_nasdaq_scan()]] - code - app.py
- [[us_market.py]] - code - us_market.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Cluster_0
SORT file.name ASC
```

## Connections to other communities
- 4 edges to [[_COMMUNITY_Cluster 4]]
- 1 edge to [[_COMMUNITY_Cluster 7]]
- 1 edge to [[_COMMUNITY_Cluster 12]]

## Top bridge nodes
- [[run_nasdaq_scan()]] - degree 4, connects to 2 communities
- [[get_nasdaq100_symbols()]] - degree 5, connects to 1 community
- [[api_index_chart()]] - degree 3, connects to 1 community
- [[api_sectors()]] - degree 3, connects to 1 community
- [[api_status()]] - degree 2, connects to 1 community