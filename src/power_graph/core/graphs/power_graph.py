import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import scipy
from typing import Any, Dict, Optional
from power_graph.core.groups import CyclicGroup, GLGroup
from power_graph.utils.power_checker import PowerChecker
from collections import deque

class PowerGraph:
    """
    Represents the power graph of a finite group (CyclicGroup or GLGroup).

    The power graph connects elements x and y if one is a positive power of the other.
    Supports both directed and undirected graphs.

    Attributes:
        group (CyclicGroup or GLGroup): The group for which the power graph is constructed.
        directed (bool): Whether the graph is directed.
        graph (networkx.Graph or networkx.DiGraph): NetworkX graph storing power relations.
        _node_map (dict): Mapping from hashable node keys to group elements.
    """

    # ---------------------- Public methods ---------------------- #
    def __init__(self, group: Any, directed: bool = False) -> None:
        """
        Initialize the PowerGraph.

        Args:
            group (CyclicGroup or GLGroup): The finite group to represent.
            directed (bool, optional): If True, build a directed power graph. Defaults to False.

        Raises:
            TypeError: If the group type is not supported.
        """
        if not isinstance(group, (CyclicGroup, GLGroup)):
            raise TypeError("Only CyclicGroup and GLGroup are supported.")
        self.group = group
        self.directed = directed
        self.graph = nx.DiGraph() if directed else nx.Graph()
        self._node_map: Dict[Any, Any] = {}
        self._build_graph()
        
    def draw(self, ax=None, title: Optional[str] = None,
         node_color: str = 'lightblue', node_size: int = 800, **kwargs) -> None:
        
        """
        Draw the power graph.

        Nodes are labeled with natural numbers. Legend is shown for CyclicGroup or GLGroup
        with ≤10 nodes. Matrices and cycles are formatted for readability.

        Args:
            title (str, optional): Title of the plot.
            node_color (str, optional): Color of nodes. Defaults to 'lightblue'.
            node_size (int, optional): Size of nodes. Defaults to 800.
            **kwargs: Additional keyword arguments passed to NetworkX drawing functions.

        Returns:
            None
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (10, 7)))

        nodes = list(self.graph.nodes)
        node_labels = {node: i + 1 for i, node in enumerate(nodes)}

        pos = kwargs.pop("pos", nx.spring_layout(self.graph, seed=42))
        nx.draw(self.graph, pos=pos, labels=node_labels, node_color=node_color,
                node_size=node_size, ax=ax, **kwargs)

        # Legend
        show_legend = len(nodes) <= 10 or isinstance(self.group, CyclicGroup)
        if show_legend and self._node_map is not None:
            legend_texts = []
            for node, number in node_labels.items():
                elem = self._node_map[node]
                if isinstance(self.group, CyclicGroup):
                    if hasattr(elem, "cyclic_form") and elem.cyclic_form:
                        cycles = ["(" + " ".join(str(i + 1) for i in cycle) + ")" 
                                for cycle in elem.cyclic_form]
                        label_str = "".join(cycles)
                    else:
                        label_str = "()"
                else:
                    label_str = '\n'.join(' '.join(str(int(x)) for x in row) for row in elem)

                patch = mpatches.Patch(facecolor='none', edgecolor='none',
                       label=f"{number}: {label_str}")
                legend_texts.append(patch)
            ax.legend(handles=legend_texts, loc='upper right', frameon=False, fontsize=8)

        if title:
            ax.set_title(title)
        ax.axis('off')

    def get_node_mapping(self) -> Dict[Any, Any]:
        """
        Get a copy of the internal node-to-element mapping.

        Returns:
            dict: Mapping from node keys to group elements.
        """
        return self._node_map.copy()

    def summarize(self) -> Dict[str, Any]:
        """
        Return a summary of the graph with statistics and adjacency/incidence matrices.

        Returns:
            dict: Contains the summary DataFrame, adjacency matrix, and incidence matrix.
        """
        G = self.graph
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()

        if G.is_directed():
            num_components = nx.number_weakly_connected_components(G)
            degrees = [d for n, d in G.degree()]
        else:
            num_components = nx.number_connected_components(G)
            degrees = [d for n, d in G.degree()]

        grado_max = max(degrees) if degrees else 0
        grado_min = min(degrees) if degrees else 0

        adjacency = nx.to_numpy_array(G, nodelist=list(G.nodes), dtype=int)
        incidence = nx.incidence_matrix(G, nodelist=list(G.nodes), oriented=self.directed).toarray()

        df = pd.DataFrame([{
            "Nodos": num_nodes,
            "Aristas": num_edges,
            "Partes conexas": num_components,
            "Grado máximo": grado_max,
            "Grado mínimo": grado_min
        }])

        return {
            "summary": df,
            "adjacency_matrix": adjacency,
            "incidence_matrix": incidence
        }

    def __repr__(self) -> str:
        """
        String representation of the PowerGraph.

        Returns:
            str: Describes type, group, number of nodes and edges.
        """
        graph_type = "Directed" if self.directed else "Undirected"
        return (f"PowerGraph({graph_type}, Group={type(self.group).__name__}, "
                f"Nodes={self.graph.number_of_nodes()}, Edges={self.graph.number_of_edges()})")

    # ---------------------- Private methods ---------------------- #
    def _get_elements(self):
        """Return all elements of the group."""
        if isinstance(self.group, CyclicGroup):
            return self.group.get_elements()
        elif isinstance(self.group, GLGroup):
            return list(self.group.bfs_generate())
        return []

    def _make_hashable(self, element: Any) -> Any:
        """Convert a group element to a hashable representation for NetworkX."""
        if isinstance(self.group, GLGroup):
            return tuple(map(tuple, element))
        return element

    def _build_graph(self) -> None:
        """
        Construct the power graph by adding nodes and edges based on power relations.

        Optimizations:
        - For GLGroup with many elements (>10), uses BFS from identity and generators.
        - For CyclicGroup or small groups, uses PowerChecker for all pairs.
        """
        elements = self._get_elements()
        self._node_map = {self._make_hashable(e): e for e in elements}
        self.graph.add_nodes_from(self._node_map.keys())
        nodes = list(self._node_map.keys())

        use_bfs = isinstance(self.group, GLGroup) and len(nodes) > 10

        if use_bfs:
            seen = set()
            queue = deque([self.group.identity])
            seen.add(self._make_hashable(self.group.identity))

            while queue:
                current = queue.popleft()
                current_key = self._make_hashable(current)

                for g in self.group.elements:
                    new_elem = self.group.multiply(current, g)
                    new_key = self._make_hashable(new_elem)

                    if new_key not in seen:
                        seen.add(new_key)
                        self.graph.add_edge(current_key, new_key)
                        queue.append(new_elem)
        else:
            if self.directed:
                for x in nodes:
                    for y in nodes:
                        if x != y and PowerChecker(self._node_map[x], self._node_map[y], self.group):
                            self.graph.add_edge(x, y)
            else:
                for i, x in enumerate(nodes):
                    for y in nodes[i + 1:]:
                        if (PowerChecker(self._node_map[x], self._node_map[y], self.group) or
                            PowerChecker(self._node_map[y], self._node_map[x], self.group)):
                            self.graph.add_edge(x, y)
