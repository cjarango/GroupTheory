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
        if isinstance(group, CyclicGroup):
            return cls._is_power_cyclic(x, y, group)
        elif isinstance(group, GLGroup):
            return cls._is_power_matrix(x, y, group)
        else:
            raise TypeError("Unsupported group type")

    # --------------------- Cyclic Group --------------------- #
    @classmethod
    def _is_power_cyclic(cls, x: Any, y: Any, group: CyclicGroup) -> bool:
        identity = group.get_identity()
        if y == identity:
            return True
        if x == identity:
            return y == identity
        
        order = group.get_element_order(x)
        current_power = x
        for k in range(1, order):
            if current_power == y:
                return True
            current_power = group.multiply(current_power, x)
        return False

    # --------------------- GLGroup --------------------- #
    @classmethod
    def _binary_exponentiation(cls, x: np.ndarray, k: int, group: GLGroup) -> np.ndarray:
        """
        Compute x^k using binary exponentiation in GLGroup.
        """
        n = group.get_dimension()
        p = group.get_modulus()
        result = np.eye(n, dtype=int) % p
        base = x.copy()

        while k > 0:
            if k % 2 == 1:
                result = group.multiply(result, base)
            base = group.multiply(base, base)
            k //= 2

        return result

    @classmethod
    def _is_power_matrix(cls, x: np.ndarray, y: np.ndarray, group: GLGroup) -> bool:
        n = group.get_dimension()
        p = group.get_modulus()
        identity = np.eye(n, dtype=int) % p

        if np.array_equal(x, y) or np.array_equal(y, identity):
            return True

        order_x = group.get_element_order(x)
        if order_x is None:
            return False

        # Check all powers from 1 to order_x using binary exponentiation
        for k in range(1, order_x + 1):
            if np.array_equal(cls._binary_exponentiation(x, k, group), y):
                return True

        return False