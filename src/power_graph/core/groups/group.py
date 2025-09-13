from abc import ABC, abstractmethod
from typing import Any, List, Dict

class Group(ABC):
    """
    Abstract base class representing a finite group.
    
    This class defines a common interface for different types of groups
    (e.g., cyclic groups, general linear groups), ensuring consistent
    access to elements, operations, and group properties.
    
    Attributes
    ----------
    n : int | None
        The order (size) of the group. Should be set by subclasses.
    elements : List[Any]
        List containing all elements of the group. Should be set by subclasses.
    identity : Any
        The identity element of the group. Should be set by subclasses.
    """
    
    def __init__(self) -> None:
        """
        Initializes the base group attributes.
        Subclasses are expected to populate `n`, `elements`, and `identity`.
        """
        self.n: int | None = None          # Group order
        self.elements: List[Any] = []      # List of group elements
        self.identity: Any = None          # Identity element of the group

    @abstractmethod
    def get_elements(self) -> List[Any]:
        """
        Returns a list of all elements in the group.

        Returns
        -------
        List[Any]
            All elements of the group.
        """
        pass
    
    @abstractmethod
    def get_identity(self) -> Any:
        """
        Returns the identity element of the group.

        Returns
        -------
        Any
            The identity element.
        """
        pass
    
    @abstractmethod
    def get_element_order(self, a: Any) -> int:
        """
        Returns the order of a given element.

        Parameters
        ----------
        a : Any
            A group element.

        Returns
        -------
        int
            The order of the element `a`.
        """
        pass
    
    @abstractmethod
    def multiply(self, a: Any, b: Any) -> Any:
        """
        Returns the product of two elements in the group.

        Parameters
        ----------
        a : Any
            A group element.
        b : Any
            A group element.

        Returns
        -------
        Any
            The product `a * b` in the group.
        """
        pass
    
    @abstractmethod
    def get_order(self) -> int:
        """
        Returns the order (number of elements) of the group.

        Returns
        -------
        int
            The order of the group.
        """
        pass
    
    @abstractmethod
    def get_element_labels(self) -> Dict[Any, str]:
        """
        Returns a mapping of elements to human-readable labels.
        Useful for visualization or debugging.

        Returns
        -------
        Dict[Any, str]
            A dictionary mapping each element to a string label.
        """
        pass
    
    @abstractmethod
    def get_generators(self) -> List[Any]:
        """
        Returns a generating set for the group.
        
        Returns
        -------
        List[Any]
            A list of group elements that generate the group.
        """
        pass
