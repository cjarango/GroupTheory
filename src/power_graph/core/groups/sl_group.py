from power_graph.core.groups import GLGroup
import numpy as np
from sympy import isprime, Matrix
from typing import List


class SLGroup(GLGroup):
    def __init__(self, n: int, p: int) -> None:
        if not isprime(p):
            raise ValueError(f"Modulus p must be prime, got p={p}")

        super().__init__(n, p)

        # Generar todos los elementos posibles de GL(n,p) usando BFS completo
        all_gl_elements = list(self.bfs_generate(max_elements=self.get_full_group_order()))

        # Filtrar solo las matrices con determinante 1
        self.sl_elements = [mat for mat in all_gl_elements if self.determinant(mat) == 1]

        # Los elementos de SL son los generadores para cualquier operación posterior
        self.elements = self.sl_elements

    def determinant(self, mat: np.ndarray) -> int:
        return int(Matrix(mat.tolist()).det() % self._p)

    def get_full_group_order(self) -> int:
        """Orden de SL(n, p) = |GL(n,p)| / (p-1)"""
        gl_order = super().get_full_group_order()
        return gl_order // (self._p - 1)

    def generate_all_elements(self) -> List[np.ndarray]:
        return self.sl_elements
