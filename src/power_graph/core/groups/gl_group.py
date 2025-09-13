import numpy as np
from sympy import isprime, Matrix
from collections import deque
from typing import List, Dict, Any, Optional, Generator
from power_graph.core.groups.group import Group

class GLGroup(Group):
    """
    General Linear Group GL(n, p) using a set of generators (bases).

    Only the generators are stored in `self.elements` as NumPy arrays.
    Full group traversal can be done dynamically when needed.

    Attributes
    ----------
    _n : int
        Dimension of the square matrices.
    _p : int
        Prime modulus for matrix entries.
    identity : np.ndarray
        Identity matrix of the group.
    elements : List[np.ndarray]
        Stored generators/bases of the group.
    """

    def __init__(self, n: int, p: int, generators: Optional[List[np.ndarray]] = None) -> None:
        """
        Initialize a GL(n, p) group.

        Parameters
        ----------
        n : int
            Dimension of the square matrices.
        p : int
            Prime modulus for matrix entries.
        generators : Optional[List[np.ndarray]], default=None
            List of generator matrices. If None, standard generators are used.
        """
        if not isprime(p):
            raise ValueError(f"Modulus p must be prime, got p={p}")

        super().__init__()
        self._n: int = n
        self._p: int = p
        self.identity: np.ndarray = np.eye(n, dtype=int)

        if generators:
            self.elements: List[np.ndarray] = [np.array(g, dtype=int) % p for g in generators]
        else:
            self.elements: List[np.ndarray] = self._generate_standard_generators()

    # --------------------- Internal generators --------------------- #

    def _generate_standard_generators(self) -> List[np.ndarray]:
        """
        Generate a standard set of generators for GL(n, p).

        Returns
        -------
        List[np.ndarray]
            List of generator matrices.
        """
        generators: List[np.ndarray] = [self.identity.copy()]

        # Transvection matrices T_ij(1)
        for i in range(self._n):
            for j in range(self._n):
                if i != j:
                    mat = np.eye(self._n, dtype=int)
                    mat[i, j] = 1
                    generators.append(mat)

        # Dilation matrices D_i(alpha) for alpha in {2,...,p-1}
        for i in range(self._n):
            for alpha in range(2, self._p):
                mat = np.eye(self._n, dtype=int)
                mat[i, i] = alpha % self._p
                generators.append(mat)

        return generators

    def _is_invertible(self, mat: np.ndarray) -> bool:
        """
        Check if a matrix is invertible modulo p.

        Parameters
        ----------
        mat : np.ndarray
            Matrix to check.

        Returns
        -------
        bool
            True if invertible, False otherwise.
        """
        try:
            det = int(round(np.linalg.det(mat))) % self._p
            return det != 0
        except:
            return False

    def _hashable_matrix(self, mat: np.ndarray) -> tuple:
        """
        Convert a NumPy matrix to a hashable tuple of tuples.

        Parameters
        ----------
        mat : np.ndarray
            Matrix to convert.

        Returns
        -------
        tuple
            Hashable representation of the matrix.
        """
        return tuple(map(tuple, mat))

    # --------------------- Getters --------------------- #

    def get_modulus(self) -> int:
        """Return the prime modulus p of the group."""
        return self._p

    def get_dimension(self) -> int:
        """Return the dimension n of the square matrices."""
        return self._n

    def get_elements(self) -> List[np.ndarray]:
        """Return the stored generators (bases) of the group."""
        return self.elements

    def get_identity(self) -> np.ndarray:
        """Return the identity matrix of GL(n, p)."""
        return self.identity

    def get_order(self) -> int:
        """Return the number of stored generators (not full group order)."""
        return len(self.elements)

    def get_full_group_order(self) -> int:
        """Return the theoretical order of the full GL(n, p) group."""
        order = 1
        for i in range(self._n):
            order *= (self._p**self._n - self._p**i)
        return order

    def get_element_labels(self) -> Dict[tuple, np.ndarray]:
        """
        Return a mapping from hashable tuples to matrix elements.

        Returns
        -------
        Dict[tuple, np.ndarray]
            Dictionary mapping hashable matrices to their NumPy representation.
        """
        return {self._hashable_matrix(mat): mat for mat in self.elements}

    # --------------------- Operations --------------------- #

    def multiply(self, a: np.ndarray, b: np.ndarray, exponent: Optional[int] = None) -> np.ndarray:
        """
        Multiply two matrices or compute a matrix power modulo p.

        Parameters
        ----------
        a : np.ndarray
            First matrix.
        b : np.ndarray
            Second matrix.
        exponent : Optional[int], default=None
            If provided, computes a^exponent instead of a*b.

        Returns
        -------
        np.ndarray
            Resulting matrix modulo p.
        """
        if exponent is not None:
            result = np.eye(a.shape[0], dtype=int)
            base = a.copy()
            k = exponent
            while k > 0:
                if k % 2 == 1:
                    result = np.matmul(result, base) % self._p
                base = np.matmul(base, base) % self._p
                k //= 2
            return result
        else:
            return np.matmul(a, b) % self._p

    def inverse(self, a: np.ndarray) -> np.ndarray:
        """
        Compute the inverse of a matrix modulo p.

        Parameters
        ----------
        a : np.ndarray
            Matrix to invert.

        Returns
        -------
        np.ndarray
            Inverse of the matrix modulo p.
        """
        return np.array(Matrix(a.tolist()).inv_mod(self._p), dtype=int) % self._p

    def get_element_order(self, a: np.ndarray) -> Optional[int]:
        """
        Return the order of a matrix element in GL(n, p).

        Parameters
        ----------
        a : np.ndarray
            Matrix element.

        Returns
        -------
        Optional[int]
            Order of the element if found, else None.
        """
        power = a.copy()
        max_order = self.get_full_group_order()
        for k in range(1, max_order + 1):
            if np.array_equal(power, self.identity):
                return k
            power = self.multiply(power, a)
        return None

    # --------------------- BFS traversal --------------------- #

    def bfs_generate(self) -> Generator[np.ndarray, None, None]:
        """
        Dynamically traverse all elements generated by self.elements using BFS.

        Yields
        ------
        np.ndarray
            Elements of the group in NumPy array form.
        """
        seen = {self._hashable_matrix(self.identity)}
        queue = deque([self.identity.copy()])

        while queue:
            current = queue.popleft()
            yield current

            for generator in self.elements:
                new_elem = self.multiply(current, generator)
                key = self._hashable_matrix(new_elem)

                if key not in seen:
                    seen.add(key)
                    queue.append(new_elem)

                    # Safety limit for very large groups
                    if len(seen) > 10000:
                        return

    def generate_all_elements(self, max_elements: Optional[int] = None) -> List[np.ndarray]:
        """
        Generate all elements of the group.

        Parameters
        ----------
        max_elements : Optional[int], default=None
            Maximum number of elements to generate (safety limit).

        Returns
        -------
        List[np.ndarray]
            List of group elements as NumPy arrays.
        """
        if max_elements is None:
            max_elements = self.get_full_group_order()

        elements: List[np.ndarray] = []
        for elem in self.bfs_generate():
            elements.append(elem)
            if len(elements) >= max_elements:
                break

        return elements

    # --------------------- Print elements --------------------- #

    def print_elements(self, max_bases: int = 10) -> None:
        """
        Print elements of the GL(n, p) group.

        - If the total number of elements is <= 10, print all elements.
        - If the total number of elements is > 10, print only the stored
          generator matrices (bases), up to `max_bases`.

        Parameters
        ----------
        max_bases : int, default=10
            Maximum number of bases to display if the group is large.
        """
        total_elements = self.get_full_group_order()

        if total_elements <= 10:
            print("All elements of the group:")
            for i, elem in enumerate(self.generate_all_elements(), start=1):
                print(f"{i}:\n{elem}\n")
        else:
            print(f"Group has {total_elements} elements (>10), printing stored generators (up to {max_bases}):\n")
            display_count = min(len(self.elements), max_bases)
            for i in range(display_count):
                print(f"Generator {i+1}:\n{self.elements[i]}\n")
            if len(self.elements) > max_bases:
                print(f"... ({len(self.elements) - max_bases} more generators not shown)")
                
    
    def get_generators(self) -> List[Any]:
        """
        Returns a generating set for the group.
        
        Returns
        -------
        List[Any]
            A list of group elements that generate the group.
        """
        raise NotImplementedError("get_generators is not implemented yet.")

    # --------------------- Representation --------------------- #

    def __repr__(self) -> str:
        full_order = self.get_full_group_order()
        return f"GLGroup(GL({self._n}, {self._p}), generators={len(self.elements)}, full_order={full_order})"
