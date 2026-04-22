---
type: community
cohesion: 0.40
members: 5
---

# Cluster 11

**Cohesion:** 0.40 - moderately connected
**Members:** 5 nodes

## Members
- [[Build NIFTY 500 scan list from index_data. Each stock mapped to a Yahoo Finance]] - rationale - app.py
- [[api_reconnect()]] - code - app.py
- [[api_reset()]] - code - app.py
- [[api_trigger_scan()]] - code - app.py
- [[load_instrument_list()]] - code - app.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Cluster_11
SORT file.name ASC
```

## Connections to other communities
- 4 edges to [[_COMMUNITY_Cluster 4]]
- 1 edge to [[_COMMUNITY_Cluster 12]]
- 1 edge to [[_COMMUNITY_Cluster 3]]

## Top bridge nodes
- [[load_instrument_list()]] - degree 7, connects to 3 communities
- [[api_reconnect()]] - degree 2, connects to 1 community
- [[api_reset()]] - degree 2, connects to 1 community
- [[api_trigger_scan()]] - degree 2, connects to 1 community