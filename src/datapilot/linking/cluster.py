"""Union-find clustering over a sparse pair-match graph.

Records are nodes, high-confidence pairs are edges; a cluster is a
connected component. Implementation uses numpy arrays so even a few
million edges resolve in under a second.
"""

from __future__ import annotations

import numpy as np


def cluster_from_pairs(
    record_ids: np.ndarray, edges: np.ndarray
) -> dict[object, int]:
    """Return ``{record_id -> cluster_id}``.

    Args:
        record_ids: 1d array of all record ids (unique).
        edges: 2d array of shape ``(n_edges, 2)`` carrying pairs of
            record ids that should be unified.

    Returns:
        Dict mapping each input id to a small integer cluster id.
        Ids that have no edges get a singleton cluster.
    """
    n = len(record_ids)
    index = {rid: i for i, rid in enumerate(record_ids.tolist())}
    parent = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        # iterative path compression — safe for huge components
        root = x
        while parent[root] != root:
            root = int(parent[root])
        while parent[x] != root:
            parent[x], x = root, int(parent[x])
        return root

    for a, b in edges:
        ia = index.get(a)
        ib = index.get(b)
        if ia is None or ib is None:
            continue
        ra, rb = find(ia), find(ib)
        if ra != rb:
            # union by simple attach; components stay shallow thanks
            # to the path compression above
            parent[ra] = rb

    # compact the parent labels into contiguous 0..k cluster ids
    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    _, cluster_ids = np.unique(roots, return_inverse=True)
    return {rid: int(cluster_ids[i]) for rid, i in index.items()}
