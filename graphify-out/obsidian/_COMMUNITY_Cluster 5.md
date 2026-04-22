---
type: community
cohesion: 0.24
members: 11
---

# Cluster 5

**Cohesion:** 0.24 - loosely connected
**Members:** 11 nodes

## Members
- [[Clear all scan results and reset to fresh state. Optionally re-login.]] - rationale - test_login.py
- [[Download NSE instruments from Angel One, filter to NIFTY 500 universe only.]] - rationale - test_login.py
- [[Run local.py]] - code - Run local.py
- [[Try to re-login and re-download instruments. Called from frontend.]] - rationale - test_login.py
- [[api_reconnect()_1]] - code - test_login.py
- [[api_reset()_1]] - code - test_login.py
- [[api_trigger_scan()_1]] - code - test_login.py
- [[create_session()]] - code - test_login.py
- [[fetch_instrument_list()]] - code - test_login.py
- [[initialize()_1]] - code - test_login.py
- [[main()]] - code - Run local.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Cluster_5
SORT file.name ASC
```

## Connections to other communities
- 7 edges to [[_COMMUNITY_Cluster 8]]
- 2 edges to [[_COMMUNITY_Cluster 9]]
- 1 edge to [[_COMMUNITY_Cluster 3]]

## Top bridge nodes
- [[create_session()]] - degree 7, connects to 2 communities
- [[fetch_instrument_list()]] - degree 7, connects to 2 communities
- [[initialize()_1]] - degree 5, connects to 1 community
- [[api_reconnect()_1]] - degree 4, connects to 1 community
- [[api_reset()_1]] - degree 4, connects to 1 community