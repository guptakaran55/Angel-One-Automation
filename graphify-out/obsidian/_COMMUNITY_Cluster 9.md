---
type: community
cohesion: 0.33
members: 7
---

# Cluster 9

**Cohesion:** 0.33 - loosely connected
**Members:** 7 nodes

## Members
- [[Fetch 730 days of daily candles for proper EMA warmup. Returns DataFrame, None (]] - rationale - test_login.py
- [[Re-login if needed, with exponential backoff to avoid rate limit spiral.]] - rationale - test_login.py
- [[ensure_session()]] - code - test_login.py
- [[fetch_candle_data()_1]] - code - test_login.py
- [[fetch_portfolio()_1]] - code - test_login.py
- [[run_full_scan()_1]] - code - test_login.py
- [[scheduler()_1]] - code - test_login.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Cluster_9
SORT file.name ASC
```

## Connections to other communities
- 5 edges to [[_COMMUNITY_Cluster 8]]
- 2 edges to [[_COMMUNITY_Cluster 5]]
- 1 edge to [[_COMMUNITY_Cluster 3]]
- 1 edge to [[_COMMUNITY_Cluster 6]]

## Top bridge nodes
- [[run_full_scan()_1]] - degree 7, connects to 4 communities
- [[ensure_session()]] - degree 5, connects to 2 communities
- [[fetch_candle_data()_1]] - degree 4, connects to 1 community
- [[fetch_portfolio()_1]] - degree 3, connects to 1 community
- [[scheduler()_1]] - degree 2, connects to 1 community