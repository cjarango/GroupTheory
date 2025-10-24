from itertools import permutations
from typing import Any, Optional
from power_graph.core.groups import SymmetricGroup, DihedralGroup
from power_graph.core.graphs.group_graph import GroupGraph


class EngelGraph(GroupGraph):
    """
    Engel graph of a finite group.
    Vertices: elements of V = G \ Z∞(G).
    There is a directed edge from x to y iff [x, n y] = 1 for some n ≥ 1.
    """

    def __init__(self, group: Any, verbose: bool = False) -> None:
        # SOLO SymmetricGroup y DihedralGroup - NO CyclicGroup
        if not isinstance(group, (SymmetricGroup, DihedralGroup)):
            raise TypeError("Only SymmetricGroup and DihedralGroup are supported.")
        
        # Verificar que el grupo no es abeliano (Z∞(G) ≠ G)
        if group.is_abelian():
            raise ValueError("Engel graph is only defined for non-abelian groups")
            
        if len(group) <= 1:
            raise ValueError("Group must have more than one element")

        super().__init__(group, directed=True, verbose=verbose)

    # ---------------------- Private methods ---------------------- #
    def _commutator(self, x: Any, y: Any) -> Any:
        """Return the commutator [x,y] = x⁻¹y⁻¹xy."""
        x_inv = self.group.get_inverse(x)
        y_inv = self.group.get_inverse(y)
        # [x,y] = x⁻¹y⁻¹xy
        return self.group.multiply(
            x_inv,
            self.group.multiply(
                y_inv,
                self.group.multiply(x, y)
            )
        )

    def _engel_commutator(self, x: Any, y: Any, max_steps: Optional[int] = None) -> Optional[int]:
        """
        Compute minimal n such that [x, n y] = 1.
        Definition: [x, 1 y] = [x,y]
                    [x, n y] = [[x, n-1 y], y]
        """
        identity = self.group.get_identity()
        
        # Usar |G| como cota máxima por finitud
        steps = max_steps if max_steps is not None else self.group.get_order()
        
        # Caso base: [x, 1 y] = [x,y]
        current = self._commutator(x, y)

        if current == identity:
            return 1

        # Iterar para n > 1: [x, n y] = [[x, n-1 y], y]
        for n in range(2, steps + 1):
            current = self._commutator(current, y)
            if current == identity:
                return n

        return None

    def _should_add_edge(self, x: Any, y: Any) -> bool:
        """Check if there should be a directed edge from x to y."""
        # No hay auto-bucles en el grafo de Engel
        if x == y:
            return False
        
        # Verificar condición de Engel: existe n tal que [x, n y] = 1
        return self._engel_commutator(x, y) is not None

    def _build_graph(self) -> None:
        """Construct the directed Engel graph on V = G \ Z∞(G)."""
        elements = self.group.get_elements()
        hypercenter = self.group.get_hypercenter()
        
        # V = G \ Z∞(G) - elementos fuera del hipercentro
        vertices = [g for g in elements if g not in hypercenter]

        # Verificación adicional: asegurar que el grafo no es vacío
        if not vertices:
            raise ValueError(f"Engel graph is empty for {self.group} - Z∞(G) = G")

        # Configurar nodos
        self._labels = self.group.get_element_labels()
        self.graph.add_nodes_from(vertices)

        if self.verbose:
            print(f"🔗 Construyendo grafo de Engel para {self.group}")
            print(f"Vértices: {len(vertices)} elementos fuera de Z∞(G)")
            print("Calculando aristas dirigidas...")

        # Construir aristas dirigidas: x → y si [x, n y] = 1 para algún n
        edge_count = 0
        for x in vertices:
            for y in vertices:
                if x != y and self._should_add_edge(x, y):
                    self.graph.add_edge(x, y)
                    edge_count += 1

        if self.verbose:
            print(f"✅ Grafo de Engel construido: {len(vertices)} vértices, {edge_count} aristas dirigidas")

    def get_engel_condition(self, x: Any, y: Any) -> Optional[int]:
        """
        Get the minimal n such that [x, n y] = 1.
        Returns None if no such n exists within |G| steps.
        """
        return self._engel_commutator(x, y)