import numpy as np
from power_graph.core.groups import CyclicGroup, GLGroup
from typing import Any

class PowerChecker:
    """
    Utility class to check whether one element is a positive power of another
    in a finite group. Supports `CyclicGroup` and `GLGroup`.

    This class does not create instances; it returns a boolean directly
    using the `__new__` method.

    Usage
    -----
    result = PowerChecker(x, y, group)  # Returns True or False
    """

    def __new__(cls, x: Any, y: Any, group: Any) -> bool:
        """
        Directly returns True or False without creating an instance.

        Parameters
        ----------
        x : Any
            First element of the group (base).
        y : Any
            Second element of the group (candidate power).
        group : CyclicGroup or GLGroup
            Group in which the check is performed.

        Returns
        -------
        bool
            True if y is a positive power of x, False otherwise.

        Raises
        ------
        TypeError
            If the group type is not supported.
        """
        if isinstance(group, CyclicGroup):
            return cls._is_power_cyclic(x, y, group)
        elif isinstance(group, GLGroup):
            return cls._is_power_matrix(x, y, group)
        else:
            raise TypeError("Unsupported group type")

    # --------------------- Cyclic Group --------------------- #
    @classmethod
    def _is_power_cyclic(cls, x: Any, y: Any, group: CyclicGroup) -> bool:
        """
        Check if y is a positive power of x in a cyclic group.

        Parameters
        ----------
        x : Any
            Base element of the cyclic group.
        y : Any
            Candidate power element of the cyclic group.
        group : CyclicGroup
            The cyclic group in which the check is performed.

        Returns
        -------
        bool
            True if y = x^k for some positive integer k, False otherwise.
        """
        identity = group.get_identity()
        if x == y or y == identity:
            return True

        power = x
        while True:
            power = group.multiply(power, x)
            if power == y:
                return True
            if power == identity:
                break
        return False

    # --------------------- GLGroup --------------------- #
    @classmethod
    def _is_power_matrix(cls, x: np.ndarray, y: np.ndarray, group: GLGroup) -> bool:
        """
        Check if y is a positive power of x in GL(n, p).

        Uses fast exponentiation and the group’s multiplication method.

        Parameters
        ----------
        x : np.ndarray
            Base matrix in GL(n, p).
        y : np.ndarray
            Candidate power matrix in GL(n, p).
        group : GLGroup
            The GL(n, p) group in which the check is performed.

        Returns
        -------
        bool
            True if y = x^k for some positive integer k, False otherwise.
        """
        n = group.get_dimension()
        p = group.get_modulus()
        identity = np.eye(n, dtype=int) % p

        # trivial cases
        if np.array_equal(x, y) or np.array_equal(y, identity):
            return True

        # get the order of x
        order_x = group.get_element_order(x)
        if order_x is None:
            return False

        # dynamic power generation using binary exponentiation
        result = identity.copy()
        base = x.copy()
        k = order_x
        while k > 0:
            if k % 2 == 1:
                result = group.multiply(result, base)
                if np.array_equal(result, y):
                    return True
            base = group.multiply(base, base)
            k //= 2

        return False
