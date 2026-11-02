"""get_prerequisites: a hand-authored concept graph, so the same topic gives the same chain."""

from typing import Any

from langchain_core.tools import tool

# topic -> what a student needs first. Edges point backwards, toward
# foundations. Keep it acyclic.
_PREREQUISITES: dict[str, tuple[str, ...]] = {
    "red-black tree": ("binary search tree", "tree rotation"),
    "avl tree": ("binary search tree", "tree rotation"),
    "b-tree": ("binary search tree",),
    "binary search tree": ("binary tree", "binary search"),
    "tree rotation": ("binary tree",),
    "binary tree": ("tree",),
    "heap": ("binary tree", "array"),
    "priority queue": ("heap",),
    "binary search": ("array", "invariant"),
    "bfs": ("graph traversal", "queue"),
    "dfs": ("graph traversal", "stack", "recursion"),
    "dijkstra": ("graph traversal", "priority queue"),
    "graph traversal": ("graph", "visited set"),
    "dynamic programming": ("recursion", "memoization"),
    "memoization": ("recursion", "hash table"),
    "quicksort": ("partitioning", "recursion"),
    "mergesort": ("recursion", "merging sorted runs"),
    "heapsort": ("heap",),
    "hash table": ("hashing", "array", "collision resolution"),
    "deadlock": ("mutex", "concurrency"),
    "mutex": ("concurrency",),
    "semaphore": ("concurrency", "mutex"),
    "virtual memory": ("paging", "address translation"),
    "paging": ("memory hierarchy",),
    "normalization": ("relational model", "functional dependency"),
    "sql index": ("b-tree", "relational model"),
    "tcp": ("packet switching", "reliable delivery"),
    "http": ("tcp",),
}

MAX_DEPTH = 6


def chain_for(topic: str) -> dict[str, Any]:
    """Prerequisites for a topic, foundations first.

    Returns an empty chain and ``known: False`` for an unrecognised topic. The
    agent's risk boundary is "never state a course fact you cannot support", so
    an honest miss is the useful answer -- a plausible invented chain is the
    failure this tool exists to prevent.
    """
    key = topic.strip().lower()
    if key not in _PREREQUISITES:
        return {"topic": topic, "known": False, "immediate": [], "chain": []}

    # Breadth-first, recording the deepest level at which each topic appears, so
    # a concept reachable by two paths is taught at the earlier point.
    depth_of: dict[str, int] = {}
    frontier = [(key, 0)]
    while frontier:
        current, depth = frontier.pop(0)
        if depth >= MAX_DEPTH:
            continue
        for parent in _PREREQUISITES.get(current, ()):
            if depth_of.get(parent, -1) >= depth + 1:
                continue
            depth_of[parent] = depth + 1
            frontier.append((parent, depth + 1))

    # Deepest first: that is teaching order. Ties sort by name so the chain is
    # reproducible rather than dependent on insertion order.
    ordered = sorted(depth_of, key=lambda name: (-depth_of[name], name))
    return {
        "topic": topic,
        "known": True,
        "immediate": sorted(_PREREQUISITES[key]),
        "chain": ordered,
    }


@tool
def get_prerequisites(topic: str) -> dict[str, Any]:
    """Get what a student needs before a topic, foundations first. known=false means the topic
    is not in the graph, so say so rather than inventing a chain."""
    return chain_for(topic)
