# Graph traversal

Graph traversal visits every node reachable from a starting node. A visited set stops the search from processing a node twice, which also keeps it from looping forever on a cycle.

## Breadth-first search

Breadth-first search (BFS) keeps its frontier in a queue, so it visits nodes in order of their distance from the start, counted in edges. On an unweighted graph, the first time BFS reaches a node is along a shortest path to it.

## Depth-first search

Depth-first search (DFS) keeps its frontier in a stack, or uses the call stack through recursion. It follows one path as far as it can before backing up.

## Cost

With an adjacency list, both visit each node once and each edge a constant number of times, so both take O(V + E) time and O(V) extra space.
