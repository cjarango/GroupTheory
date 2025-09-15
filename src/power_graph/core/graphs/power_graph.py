import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, List, Set
from power_graph.core.groups import CyclicGroup, GLGroup
from power_graph.utils.power_checker import PowerChecker
from collections import deque

class PowerGraph:
    """
    Represents the power graph of a finite group (CyclicGroup or GLGroup).
    Connects elements x and y if one is a positive power of the other.
    Supports directed and undirected graphs.
    """
    
    def __init__(self, group: Any, directed: bool = False) -> None:
        if not isinstance(group, (CyclicGroup, GLGroup)):
            raise TypeError("Only CyclicGroup and GLGroup are supported.")
        self.group = group
        self.directed = directed
        self.graph = nx.DiGraph() if directed else nx.Graph()
        self._node_map: Dict[Any, Any] = {}
        self._build_graph()
    
    # ---------------------- Public methods ---------------------- #
    def draw(self, ax=None, title: Optional[str] = None,
             node_color: str = 'lightblue', node_size: int = 800,
             with_labels: bool = True, with_legend: bool = True,
             show: bool = True, **kwargs) -> Optional[plt.Figure]:
        fig = None
        created_new_figure = False
        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (10, 7)))
            created_new_figure = True
        
        nodes = list(self.graph.nodes)
        node_labels = {node: i + 1 for i, node in enumerate(nodes)}
        pos = kwargs.pop("pos", nx.spring_layout(self.graph, seed=42))
        
        nx.draw(self.graph, pos=pos,
                labels=node_labels if with_labels else None,
                node_color=node_color,
                node_size=node_size,
                ax=ax,
                **kwargs)
        
        if with_legend and self._node_map is not None:
            show_legend = len(nodes) <= 10 or isinstance(self.group, CyclicGroup)
            if show_legend:
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
                        try:
                            label_str = '\n'.join(' '.join(str(int(x)) for x in row) for row in elem)
                        except:
                            label_str = str(elem)
                    patch = mpatches.Patch(facecolor='none', edgecolor='none',
                                            label=f"{number}: {label_str}")
                    legend_texts.append(patch)
                ax.legend(handles=legend_texts, loc='upper right', frameon=False, fontsize=8)
        
        if title:
            ax.set_title(title)
        ax.axis('off')
        if show and created_new_figure:
            plt.show()
        return fig if created_new_figure else None
    
    def get_node_mapping(self) -> Dict[Any, Any]:
        return self._node_map.copy()
    
    def get_node_order(self) -> List[Any]:
        nodes = list(self.graph.nodes())
        try:
            return sorted(nodes)
        except TypeError:
            return sorted(nodes, key=str)
    
    def get_adjacency_matrix(self) -> np.ndarray:
        node_list = self.get_node_order()
        return nx.to_numpy_array(self.graph, nodelist=node_list, dtype=int)
    
    def get_incidence_matrix(self) -> np.ndarray:
        node_list = self.get_node_order()
        if self.graph.is_directed():
            return nx.incidence_matrix(self.graph, nodelist=node_list, oriented=True).toarray()
        else:
            return nx.incidence_matrix(self.graph, nodelist=node_list, oriented=False).toarray()
    
    def summarize(self) -> Dict[str, Any]:
        G = self.graph
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        node_list = self.get_node_order()
        
        if G.is_directed():
            num_components = nx.number_weakly_connected_components(G)
            in_degrees = [d for n, d in G.in_degree()]
            out_degrees = [d for n, d in G.out_degree()]
            degrees = [d for n, d in G.degree()]
        else:
            num_components = nx.number_connected_components(G)
            degrees = [d for n, d in G.degree()]
        
        grado_max = max(degrees) if degrees else 0
        grado_min = min(degrees) if degrees else 0
        grado_promedio = sum(degrees) / len(degrees) if degrees else 0
        
        try:
            adjacency = self.get_adjacency_matrix()
            incidence = self.get_incidence_matrix()
            matrices_available = True
        except Exception as e:
            adjacency = np.array([])
            incidence = np.array([])
            matrices_available = False
        
        stats_data = {
            "Nodos": num_nodes,
            "Aristas": num_edges,
            "Partes conexas": num_components,
            "Grado máximo": grado_max,
            "Grado mínimo": grado_min,
            "Grado promedio": round(grado_promedio, 2)
        }
        
        if G.is_directed():
            stats_data.update({
                "Grado entrada máximo": max(in_degrees) if in_degrees else 0,
                "Grado salida máximo": max(out_degrees) if out_degrees else 0,
                "Grado entrada mínimo": min(in_degrees) if in_degrees else 0,
                "Grado salida mínimo": min(out_degrees) if in_degrees else 0
            })
        
        df = pd.DataFrame([stats_data])
        
        return {
            "summary": df,
            "adjacency_matrix": adjacency,
            "incidence_matrix": incidence,
            "node_order": node_list,
            "matrices_available": matrices_available
        }
    
    def verify_matrices(self) -> Dict[str, bool]:
        node_list = self.get_node_order()
        n = len(node_list)
        m = self.graph.number_of_edges()
        adjacency = self.get_adjacency_matrix()
        incidence = self.get_incidence_matrix()
        
        adjacency_correct = adjacency.shape == (n, n)
        incidence_correct = incidence.shape == (n, m)
        
        symmetric = True
        if not self.graph.is_directed():
            symmetric = np.array_equal(adjacency, adjacency.T)
        
        calculated_degrees = adjacency.sum(axis=1 if self.graph.is_directed() else 0)
        actual_degrees = np.array([self.graph.degree(node) for node in node_list])
        degrees_match = np.array_equal(calculated_degrees, actual_degrees)
        diagonal_zero = np.all(np.diag(adjacency) == 0)
        
        return {
            "adjacency_dimensions_correct": adjacency_correct,
            "incidence_dimensions_correct": incidence_correct,
            "adjacency_symmetric": symmetric,
            "degrees_match": degrees_match,
            "no_self_loops": diagonal_zero,
            "all_tests_passed": all([adjacency_correct, incidence_correct, symmetric if not self.graph.is_directed() else True, degrees_match, diagonal_zero])
        }
    
    def verify_graph_construction(self) -> Dict[str, Any]:
        node_list = self.get_node_order()
        n = len(node_list)
        correct_edges = 0
        incorrect_edges = 0
        missing_edges = 0
        
        for edge in self.graph.edges():
            x_key, y_key = edge
            x_elem = self._node_map[x_key]
            y_elem = self._node_map[y_key]
            if PowerChecker(x_elem, y_elem, self.group):
                correct_edges += 1
            else:
                incorrect_edges += 1
        
        if n <= 20:
            for i in range(n):
                for j in range(n):
                    if i != j:
                        x_key = node_list[i]
                        y_key = node_list[j]
                        x_elem = self._node_map[x_key]
                        y_elem = self._node_map[y_key]
                        has_edge = self.graph.has_edge(x_key, y_key)
                        should_have_edge = PowerChecker(x_elem, y_elem, self.group)
                        if should_have_edge and not has_edge:
                            missing_edges += 1
        
        return {
            "total_edges": self.graph.number_of_edges(),
            "correct_edges": correct_edges,
            "incorrect_edges": incorrect_edges,
            "missing_edges": missing_edges if n <= 20 else "Not checked (group too large)",
            "construction_correct": incorrect_edges == 0 and (n > 20 or missing_edges == 0)
        }
    
    def __repr__(self) -> str:
        graph_type = "Directed" if self.directed else "Undirected"
        return (f"PowerGraph({graph_type}, Group={type(self.group).__name__}, "
                f"Nodes={self.graph.number_of_nodes()}, Edges={self.graph.number_of_edges()})")
    
    # ---------------------- Private methods ---------------------- #
    def _get_elements(self) -> List[Any]:
        if isinstance(self.group, CyclicGroup):
            return self.group.get_elements()
        elif isinstance(self.group, GLGroup):
            return list(self.group.bfs_generate())
        return []

    def _make_hashable(self, element: Any) -> Any:
        if isinstance(self.group, GLGroup):
            return tuple(map(tuple, element))
        elif hasattr(element, '__hash__'):
            return element
        else:
            try:
                return tuple(element)
            except TypeError:
                return str(element)
    
    def _build_graph(self) -> None:
        elements = self._get_elements()
        self._node_map = {self._make_hashable(e): e for e in elements}
        self.graph.add_nodes_from(self._node_map.keys())
        nodes = list(self._node_map.keys())
        use_bfs = isinstance(self.group, GLGroup) and len(nodes) > 10
        if use_bfs:
            self._build_with_bfs()
        else:
            self._build_with_power_checker(nodes)

    def _build_with_bfs(self) -> None:
        seen: Set[Any] = set()
        queue = deque([self.group.identity])
        identity_key = self._make_hashable(self.group.identity)
        seen.add(identity_key)
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
    
    def _build_with_power_checker(self, nodes: List[Any]) -> None:
        n = len(nodes)
        if self.directed:
            for i in range(n):
                x = nodes[i]
                x_elem = self._node_map[x]
                for j in range(n):
                    if i == j:
                        continue
                    y = nodes[j]
                    y_elem = self._node_map[y]
                    if PowerChecker(x_elem, y_elem, self.group) and not self.graph.has_edge(x, y):
                        self.graph.add_edge(x, y)
        else:
            for i in range(n):
                x = nodes[i]
                x_elem = self._node_map[x]
                for j in range(n):
                    if i == j:
                        continue
                    y = nodes[j]
                    y_elem = self._node_map[y]
                    if (PowerChecker(x_elem, y_elem, self.group) or PowerChecker(y_elem, x_elem, self.group)):
                        if not self.graph.has_edge(x, y):
                            self.graph.add_edge(x, y)
