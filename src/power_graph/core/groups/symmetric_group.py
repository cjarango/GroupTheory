from typing import Any, List, Dict
from sympy.combinatorics import SymmetricGroup as SymSymmetricGroup
from sympy.combinatorics.permutations import Permutation
from sympy.combinatorics import PermutationGroup
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
        self.elements: List[Any] = list(self._sym_group.generate_dimino())
        self.identity: Any = self._sym_group.identity


    @classmethod
    def dihedral(cls, n: int) -> "SymmetricGroup":
        """
        Constructor alternativo para crear el grupo diedral D_n.
        Contiene rotaciones y reflexiones del polígono regular de n lados.
        """
        if n < 2:
            raise ValueError("El orden del polígono debe ser al menos 2.")

        # Rotación generadora
        rotation = Permutation(list(range(1, n)) + [0])  # (0 1 2 ... n-1)
        # Reflexión: intercambiar 0 con n-1, 1 con n-2, etc.
        reflection_list = list(range(n))
        for i in range(n // 2):
            reflection_list[i], reflection_list[n - 1 - i] = reflection_list[n - 1 - i], reflection_list[i]
        reflection = Permutation(reflection_list)

        # Crear el grupo diedral como subgrupo de S_n
        dihedral_group = PermutationGroup([rotation, reflection])
        elements = list(dihedral_group.generate_dimino())

        # Crear la instancia
        instance = cls.__new__(cls)
        instance.n = n
        instance._sym_group = dihedral_group
        instance.elements = elements
        instance.identity = dihedral_group.identity
        return instance

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
        labels: Dict[Any, str] = {}
        for el in self.elements:
            if not el.cyclic_form:
                labels[el] = "()"
            else:
                # Formatear la notación cíclica correctamente
                cycles = []
                for cycle in el.cyclic_form:
                    cycles.append(f"({' '.join(map(str, cycle))})")
                labels[el] = ''.join(cycles)
        return labels

    def print_elements(self) -> None:
        labels = self.get_element_labels()
        for i, el in enumerate(self.elements, start=1):
            print(f"{i}: {labels[el]}")

    def __repr__(self) -> str:
        return f"SymmetricGroup(S_{self.n})"

    def __len__(self) -> int:
        return self.get_order()

    def __contains__(self, item: Any) -> bool:
        return item in self.elements

    def get_generators(self) -> List[Any]:
        if hasattr(self._sym_group, 'generators'):
            return list(self._sym_group.generators)
        else:
            # Para subgrupos que no tienen generadores explícitos,
            # devolvemos algunos elementos representativos
            if len(self.elements) > 1:
                return self.elements[:min(2, len(self.elements))]
            return [self.identity]

    def get_inverse(self, a: Any) -> Any:
        """Get the inverse of an element."""
        return a ** -1

    def is_abelian(self) -> bool:
        """Check if the group is abelian."""
        return self.n <= 2  # S_n es abeliano solo para n ≤ 2

    def conjugate(self, a: Any, g: Any) -> Any:
        """Compute the conjugate g·a·g⁻¹."""
        return g * a * (g ** -1)

    def get_center(self) -> List[Any]:
        """Get the center of the group."""
        if self.n <= 2:
            return self.elements
        else:
            # El centro de S_n para n ≥ 3 es trivial
            return [self.identity]