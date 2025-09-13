from typing import Any, List, Dict
from sympy.combinatorics import CyclicGroup as SymCyclicGroup
from power_graph.core.groups.group import Group


class CyclicGroup(Group):
    """
    Concrete implementation of a cyclic group C_n.

    Inherits from the abstract 'Group' class and provides concrete
    implementations for all abstract methods. Supports element access,
    multiplication, order computation, and human-readable labels.
    """

    def __init__(self, n: int) -> None:
        """
        Initialize a cyclic group of order n.

        Parameters
        ----------
        n : int
            The order (number of elements) of the cyclic group.
        """
        super().__init__()
        self.n: int = n
        self._sym_group: SymCyclicGroup = SymCyclicGroup(n)  # internal SymPy group
        self.elements: List[Any] = list(self._sym_group.generate_dimino())
        self.identity: Any = self._sym_group.identity
        self.generator: Any = self._sym_group.generators[0]  # único generador

    def get_elements(self) -> List[Any]:
        return self.elements

    def get_identity(self) -> Any:
        return self.identity

    def get_element_order(self, a: Any) -> int:
        return a.order()

    def multiply(self, a: Any, b: Any) -> Any:
        return a * b

    def get_order(self) -> int:
        return len(self.elements)

    def get_element_labels(self) -> Dict[Any, str]:
        """
        Return a mapping of elements to human-readable labels as powers of the generator.
        """
        labels: Dict[Any, str] = {}
        for k, el in enumerate(self.elements):
            if el == self.identity:
                labels[el] = "e"
            else:
                labels[el] = f"g^{k}"
        return labels

    def print_elements(self) -> None:
        labels = self.get_element_labels()
        for i, el in enumerate(self.elements, start=1):
            print(f"{i}: {labels[el]}")

    def __repr__(self) -> str:
        return f"CyclicGroup(C_{self.n})"

    def __len__(self) -> int:
        return self.get_order()

    def __contains__(self, item: Any) -> bool:
        return item in self.elements

    def get_generators(self) -> List[Any]:
        """
        Return the canonical generator(s) of the cyclic group.
        """
        return [self.generator]
