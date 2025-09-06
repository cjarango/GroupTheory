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

    def get_elements(self) -> List[Any]:
        """Return all elements of the cyclic group."""
        return self.elements

    def get_identity(self) -> Any:
        """Return the identity element of the cyclic group."""
        return self.identity

    def get_element_order(self, a: Any) -> int:
        """Return the order of a specific element in the group."""
        return a.order()

    def multiply(self, a: Any, b: Any) -> Any:
        """Compute the product of two elements in the cyclic group."""
        return a * b

    def get_order(self) -> int:
        """Return the total number of elements in the cyclic group."""
        return len(self.elements)

    def get_element_labels(self) -> Dict[Any, str]:
        """
        Return a mapping of elements to human-readable labels in cyclic notation.
        """
        labels: Dict[Any, str] = {}
        for el in self.elements:
            if not el.cyclic_form:
                labels[el] = "()"
            else:
                labels[el] = "".join(f"({''.join(map(str, cycle))})" for cycle in el.cyclic_form)
        return labels

    def print_elements(self) -> None:
        """
        Print all elements of the cyclic group in cyclic notation.

        Each element is printed on a separate line with its index.
        """
        labels = self.get_element_labels()
        for i, el in enumerate(self.elements, start=1):
            print(f"{i}: {labels[el]}")

    def __repr__(self) -> str:
        return f"CyclicGroup(C_{self.n})"

    def __len__(self) -> int:
        return self.get_order()

    def __contains__(self, item: Any) -> bool:
        return item in self.elements
