from itertools import combinations
from typing import Any, Optional
from power_graph.core.groups import SymmetricGroup, CyclicGroup
from power_graph.core.graphs.group_graph import GroupGraph


class EngelGraph(GroupGraph):
    """
    Engel graph of a finite group.
    Vertices: elements of V = G \ Z∞(G).
    Two elements x and y are adjacent iff [x,_n y] = 1 for some n ≥ 1
    or [y,_m x] = 1 for some m ≥ 1.
    """

    def __init__(self, group: Any, verbose: bool = False) -> None:
        if not isinstance(group, (SymmetricGroup, CyclicGroup)):
            raise TypeError("Only SymmetricGroup and CyclicGroup are supported.")
        if len(group) <= 1:
            raise ValueError("Group must have more than one element")

        super().__init__(group, directed=False, verbose=verbose)

    # ---------------------- Private methods ---------------------- #
    def _commutator(self, x: Any, y: Any) -> Any:
        """Return the commutator [x,y] = xyx⁻¹y⁻¹."""  # CORREGIDO
        x_inv = self.group.get_inverse(x)
        y_inv = self.group.get_inverse(y)
        return self.group.multiply(
            x,
            self.group.multiply(
                y,
                self.group.multiply(x_inv, y_inv)
            )
        )

    def _engel_commutator(self, x: Any, y: Any, max_steps: Optional[int] = None) -> Optional[int]:
        """
        Compute minimal n such that [x, _n y] = 1.
        Definition: [x, _1 y] = [x,y]
                    [x, _n y] = [[x, _n-1 y], y]
        """
        identity = self.group.get_identity()
        steps = max_steps if max_steps is not None else len(self.group)

        # Caso base: [x, _1 y] = [x,y]
        current = self._commutator(x, y)

        if current == identity:
            return 1

        # Iterar para n > 1
        for n in range(2, steps + 1):
            current = self._commutator(current, y)
            if current == identity:
                return n

        return None

    def _is_adjacent(self, x: Any, y: Any) -> bool:
        """Check if x and y are adjacent in the Engel graph."""
        if x == y:
            return False

        if self._engel_commutator(x, y) is not None:
            return True
        if self._engel_commutator(y, x) is not None:
            return True

        return False

    def _build_graph(self) -> None:
        """Construct the Engel graph on V = G \ Z∞(G)."""
        elements = self.group.get_elements()

        # El hipercentro ya viene implementado en los grupos
        hypercenter = self.group.get_hypercenter() 
        vertices = [g for g in elements if g not in hypercenter]

        # Caso especial: hipercentro = G → grafo vacío
        if not vertices:
            if self.verbose:
                print("⚠️ El hipercentro coincide con todo el grupo.")
                print("El grafo de Engel es vacío (sin vértices ni aristas).")
            return

        self._labels = {g: str(g) for g in vertices}
        self.graph.add_nodes_from(vertices)

        if self.verbose:
            print("🔗 Construyendo aristas de Engel...")

        for x, y in combinations(vertices, 2):
            if self._is_adjacent(x, y):
                self.graph.add_edge(x, y)

        if self.verbose:
            print(f"✅ Grafo construido: {len(vertices)} nodos, {self.graph.number_of_edges()} aristas")