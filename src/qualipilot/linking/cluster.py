"""Union-find clustering over a sparse pair-match graph.

Records are nodes, high-confidence pairs are edges, and clusters are
connected components. Parent and rank arrays are stored in NumPy.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable

import numpy as np


def cluster_from_pairs(
    record_ids: Collection[object],
    edges: Iterable[tuple[object, object]],
) -> dict[object, int]:
    """Return ``{record_id -> cluster_id}``.

    Args:
        record_ids: all record ids, which must be unique.
        edges: pairs of record ids that should be unified.

    Returns:
        Dict mapping each input id to a small integer cluster id.
        Ids that have no edges get a singleton cluster.
    """
    n = len(record_ids)
    index = {
        _python_scalar(record_id): position
        for position, record_id in enumerate(record_ids)
    }
    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.uint8)

    def find(x: int) -> int:
        # iterative path compression — safe for huge components
        root = x
        while parent[root] != root:
            root = int(parent[root])
        while parent[x] != root:
            parent[x], x = root, int(parent[x])
        return root

    for a, b in edges:
        ia = index.get(_python_scalar(a))
        ib = index.get(_python_scalar(b))
        if ia is None or ib is None:
            continue
        ra, rb = find(ia), find(ib)
        if ra != rb:
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

    components: dict[int, list[object]] = {}
    for record_id, position in index.items():
        components.setdefault(find(position), []).append(record_id)
    ordered = sorted(
        components.values(),
        key=lambda members: min(_stable_id_key(member) for member in members),
    )
    return {
        record_id: cluster_id
        for cluster_id, members in enumerate(ordered)
        for record_id in members
    }


def _stable_id_key(value: object) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def _python_scalar(value: object) -> object:
    if isinstance(value, np.datetime64):
        converted = value.item()
        if not isinstance(converted, int):
            return converted
        nanoseconds = value.astype("datetime64[ns]")
        if int(nanoseconds.astype(np.int64)) % 1000:
            return value
        return nanoseconds.astype("datetime64[us]").item()
    if isinstance(value, np.timedelta64):
        converted = value.item()
        if not isinstance(converted, int):
            return converted
        nanoseconds = value.astype("timedelta64[ns]")
        if int(nanoseconds.astype(np.int64)) % 1000:
            return value
        return nanoseconds.astype("timedelta64[us]").item()
    return value.item() if isinstance(value, np.generic) else value
