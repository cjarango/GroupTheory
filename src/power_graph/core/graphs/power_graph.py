import networkx as nx
from typing import Any, List, Set
from collections import deque
from power_graph.core.groups import CyclicGroup, GLGroup
from power_graph.utils.power_checker import PowerChecker
from power_graph.core.graphs.group_graph import GroupGraph


class PowerGraph(GroupGraph):
    """
    Power graph of a finite group (CyclicGroup or GLGroup).
    Vertices: elements of G.
    Two elements x, y are adjacent iff one is a positive power of the other.
    Supports directed and undirected graphs.
    """

    def __init__(self, group: Any, directed: bool = False, verbose: bool = False) -> None:
        if not isinstance(group, (CyclicGroup, GLGroup)):
            raise TypeError("Only CyclicGroup and GLGroup are supported.")
        super().__init__(group, directed=directed, verbose=verbose)

    # ---------------------- Required overrides ---------------------- #
    def _get_elements(self) -> List[Any]:
        """Return all elements of the group."""
        if isinstance(self.group, CyclicGroup):
            return self.group.get_elements()
        elif isinstance(self.group, GLGroup):
            return list(self.group.bfs_generate())
        return []

    def _make_hashable(self, element: Any) -> Any:
        """Ensure elements are hashable for use as graph nodes."""
        if isinstance(self.group, GLGroup):
            return tuple(map(tuple, element))  # matrix → tuple of tuples
        elif hasattr(element, '__hash__'):
            return element
        else:
            try:
                return tuple(element)
            except TypeError:
                return str(element)

    def _is_adjacent(self, x: Any, y: Any) -> bool:
        """Check adjacency in the power graph."""
        return PowerChecker(x, y, self.group)

    def _build_graph(self) -> None:
        """Construct the power graph."""
        elements = self._get_elements()
        self._labels = {self._make_hashable(e): str(e) for e in elements}
        self.graph.add_nodes_from(self._labels.keys())

        nodes = list(self._labels.keys())
        use_bfs = isinstance(self.group, GLGroup) and len(nodes) > 10

        if self.verbose:
            print("🔗 Construyendo aristas de PowerGraph...")

        if use_bfs:
            self._build_with_bfs()
        else:
            self._build_with_power_checker(nodes)

        if self.verbose:
            print(f"✅ Grafo construido: {len(nodes)} nodos, {self.graph.number_of_edges()} aristas")

    # ---------------------- Private methods ---------------------- #
    def _build_with_bfs(self) -> None:
        """Optimized BFS construction for GLGroup (large groups)."""
        seen: Set[Any] = set()
        queue = deque([self.group.identity])
        identity_key = self._make_hashable(self.group.identity)
        seen.add(identity_key)

        while queue:
            current = queue.popleft()
            current_key = self._make_hashable(current)

            for g in self.group.elements:
                new_elem = self.group.multiply(current, g)
                new_key = self._make_hashable(new_elem)

                if new_key not in seen:
                    seen.add(new_key)
                    self.graph.add_edge(current_key, new_key)
                    queue.append(new_elem)

    def _build_with_power_checker(self, nodes: List[Any]) -> None:
        """Adjacency check using PowerChecker (optimized double loop)."""
        n = len(nodes)
        if self.directed:
            for i in range(n):
                x = nodes[i]
                for j in range(n):
                    if i == j:
                        continue
                    y = nodes[j]
                    if self._is_adjacent(self._labels[x], self._labels[y]):
                        self.graph.add_edge(x, y)
        else:
            for i in range(n):
                x = nodes[i]
                for j in range(i + 1, n):
                    y = nodes[j]
                    if (self._is_adjacent(self._labels[x], self._labels[y]) or
                        self._is_adjacent(self._labels[y], self._labels[x])):
                        self.graph.add_edge(x, y)
