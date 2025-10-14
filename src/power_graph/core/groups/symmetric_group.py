from typing import Any, List, Dict
from functools import lru_cache
from sympy.combinatorics import SymmetricGroup as SymSymmetricGroup
from power_graph.core.groups.group import Group


class SymmetricGroup(Group):
    """
    Concrete implementation of a symmetric group S_n.

    Inherits from the abstract 'Group' class and provides concrete
    implementations for all abstract methods. Supports element access,
    multiplication, order computation, and human-readable labels.
    """

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n: int = n
        self._sym_group: SymSymmetricGroup = SymSymmetricGroup(n)
        self.identity: Any = self._sym_group.identity

    # ----------------------------
    # Lazy evaluation of elements
    # ----------------------------
    @lru_cache(maxsize=1)
    def get_elements(self) -> List[Any]:
        return list(self._sym_group.generate_dimino())

    def get_identity(self) -> Any:
        return self.identity

    def get_element_order(self, a: Any) -> int:
        return a.order()

    def multiply(self, a: Any, b: Any) -> Any:
        return a * b

    def get_order(self) -> int:
        return len(self.get_elements())

    # ----------------------------
    # Labels / Pretty printing
    # ----------------------------
    def get_element_labels(self, one_indexed: bool = True) -> Dict[Any, str]:
        labels: Dict[Any, str] = {}
        for el in self.get_elements():
            if not el.cyclic_form:
                labels[el] = "()"
            else:
                cycles = []
                for cycle in el.cyclic_form:
                    if one_indexed:
                        cycle = [x + 1 for x in cycle]
                    cycles.append(f"({' '.join(map(str, cycle))})")
                labels[el] = ''.join(cycles)
        return labels

    def print_elements(self, one_indexed: bool = True) -> None:
        labels = self.get_element_labels(one_indexed=one_indexed)
        for i, el in enumerate(self.get_elements(), start=1):
            print(f"{i}: {labels[el]}")

    # ----------------------------
    # Structural methods
    # ----------------------------
    def __repr__(self) -> str:
        return f"SymmetricGroup(S_{self.n}, order={self.get_order()})"

    def __len__(self) -> int:
        return self.get_order()

    def __contains__(self, item: Any) -> bool:
        return item in self.get_elements()

    def get_generators(self) -> List[Any]:
        if hasattr(self._sym_group, "generators"):
            return list(self._sym_group.generators)
        elements = self.get_elements()
        if len(elements) > 1:
            return elements[:min(2, len(elements))]
        return [self.identity]

    def get_inverse(self, a: Any) -> Any:
        """Get the inverse of an element."""
        return a ** -1

    def is_abelian(self) -> bool:
        """S_n es abeliano solo si n <= 2."""
        return self.n <= 2

    def conjugate(self, a: Any, g: Any) -> Any:
        """Compute the conjugate g·a·g⁻¹."""
        return g * a * (g ** -1)

    def get_center(self) -> List[Any]:
        """
        Centro de S_n:
        - si n <= 2: todo el grupo
        - si n > 2: solo la identidad
        """
        if self.n <= 2:
            return self.get_elements()
        return [self.identity]

    def get_conjugacy_classes(self) -> List[List[Any]]:
        """
        Compute the conjugacy classes of the group.
        Uses SymPy's built-in method when available.
        """
        if hasattr(self._sym_group, "conjugacy_classes"):
            return [list(cls) for cls in self._sym_group.conjugacy_classes()]

        # fallback manual implementation
        elements = self.get_elements()
        classes: List[List[Any]] = []
        seen = set()

        for g in elements:
            if g in seen:
                continue
            conj_class = {h * g * (h ** -1) for h in elements}
            classes.append(list(conj_class))
            seen.update(conj_class)

        return classes

    def print_conjugacy_classes(self, one_indexed: bool = False) -> None:
        classes = self.get_conjugacy_classes()
        labels = self.get_element_labels(one_indexed=one_indexed)
        for i, cls in enumerate(classes, start=1):
            print(f"Class {i} (size={len(cls)}): {[labels[g] for g in cls]}")

    # ----------------------------
    # Ascending central series & hypercenter - OPTIMIZADO
    # ----------------------------
    def _compute_ascending_central_series(self) -> List[List[Any]]:
        """
        Serie central ascendente de S_n.
        Para n ≥ 3, el centro es trivial y la serie se estabiliza en Z0.
        """
        if hasattr(self, "_ascending_central_series"):
            return self._ascending_central_series

        # Z0 siempre es la identidad
        Z0 = [self.identity]
        
        if self.n <= 2:
            # S₁ y S₂ son abelianos → hipercentro = grupo completo
            series = [Z0, self.get_elements()]
        else:
            # Sₙ con n ≥ 3 tiene centro trivial → serie se estabiliza en Z0
            series = [Z0]

        self._ascending_central_series = series
        return series

    def get_hypercenter(self) -> List[Any]:
        """Devuelve el hipercentro de S_n (último término de la serie ascendente)."""
        return self._compute_ascending_central_series()[-1]

    def print_ascending_central_series(self, one_indexed: bool = False) -> None:
        """Pretty print de la serie central ascendente."""
        labels = self.get_element_labels(one_indexed=one_indexed)
        series = self._compute_ascending_central_series()
        for i, Zi in enumerate(series):
            print(f"Z{i}(G) = {[labels[g] for g in Zi]}")