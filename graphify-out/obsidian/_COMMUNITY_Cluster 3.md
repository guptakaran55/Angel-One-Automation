---
type: community
cohesion: 0.18
members: 12
---

# Cluster 3

**Cohesion:** 0.18 - loosely connected
**Members:** 12 nodes

## Members
- [[Download current NIFTY 500 from NSE using urllib.     Uses polling instead of T]] - rationale - index_data.py
- [[Fetch 2 years of daily OHLCV data from Yahoo Finance v8 chart API.     Uses dir]] - rationale - app.py
- [[Fetch holdings from Zerodha. Returns list of dicts with tradingsymbol.]] - rationale - app.py
- [[Index constituent lists for Indian stock markets. NIFTY 500 downloaded dynamica]] - rationale - index_data.py
- [[build_index_tags()]] - code - index_data.py
- [[download_nifty500()]] - code - index_data.py
- [[fetch_candle_data()]] - code - app.py
- [[fetch_portfolio()]] - code - app.py
- [[get_scan_universe()]] - code - index_data.py
- [[get_tags_for()]] - code - index_data.py
- [[index_data.py]] - code - index_data.py
- [[run_nifty_scan()]] - code - app.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Cluster_3
SORT file.name ASC
```

## Connections to other communities
- 3 edges to [[_COMMUNITY_Cluster 4]]
- 1 edge to [[_COMMUNITY_Cluster 11]]
- 1 edge to [[_COMMUNITY_Cluster 7]]
- 1 edge to [[_COMMUNITY_Cluster 5]]
- 1 edge to [[_COMMUNITY_Cluster 9]]

## Top bridge nodes
- [[run_nifty_scan()]] - degree 5, connects to 2 communities
- [[get_scan_universe()]] - degree 4, connects to 2 communities
- [[fetch_candle_data()]] - degree 3, connects to 1 community
- [[fetch_portfolio()]] - degree 3, connects to 1 community
- [[get_tags_for()]] - degree 3, connects to 1 community