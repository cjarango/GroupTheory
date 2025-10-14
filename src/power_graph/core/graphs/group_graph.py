import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, List
from abc import ABC, abstractmethod


class GroupGraph(ABC):
    """
    Clase base para grafos asociados a grupos finitos.
    Contiene funcionalidad común: construcción de matrices, resumen, dibujo.
    Las subclases deben implementar `_build_graph`.
    """

    def __init__(self, group: Any, directed: bool = False, verbose: bool = False) -> None:
        self.group = group
        self.directed = directed
        self.verbose = verbose
        self.graph = nx.DiGraph() if directed else nx.Graph()
        self._labels: Dict[Any, str] = {}
        self._node_map: Dict[Any, Any] = {}  # útil en grafos con mapping de nodos
        self._build_graph()

    # ---------------------- Abstract methods ---------------------- #
    @abstractmethod
    def _build_graph(self) -> None:
        """Construir el grafo. Cada subclase implementa su propia lógica."""
        pass

    # ---------------------- Public methods ---------------------- #
    def draw(self, ax=None, title: Optional[str] = None,
             node_color: str = 'lightblue', node_size: int = 800,
             with_labels: bool = True, with_legend: bool = True,
             **kwargs) -> Optional[plt.Figure]:
        """Dibuja el grafo con leyenda opcional."""
        if not self.graph.nodes:
            print("⚠️ Grafo vacío, nada que dibujar.")
            return None

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (10, 7)))
            fig_created = True
        else:
            fig_created = False

        nodes = list(self.graph.nodes)
        node_labels = {node: i + 1 for i, node in enumerate(nodes)}
        pos = kwargs.pop("pos", nx.spring_layout(self.graph, seed=42))

        nx.draw(self.graph, pos=pos,
                labels=node_labels if with_labels else None,
                node_color=node_color,
                node_size=node_size,
                ax=ax,
                **kwargs)

        if with_legend and self._labels:
            legend_texts = []
            for node, number in node_labels.items():
                label = self._labels.get(node, str(node))
                patch = mpatches.Patch(facecolor='none', edgecolor='none',
                                       label=f"{number}: {label}")
                legend_texts.append(patch)
            ax.legend(handles=legend_texts, loc='upper right',
                      frameon=False, fontsize=8)

        if title:
            ax.set_title(title)
        ax.axis('off')

        if fig_created:
            return fig
        return None

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
            degrees = [d for n, d in G.degree()]
        else:
            num_components = nx.number_connected_components(G)
            degrees = [d for n, d in G.degree()]

        grado_max = max(degrees) if degrees else 0
        grado_min = min(degrees) if degrees else 0
        grado_prom = sum(degrees) / len(degrees) if degrees else 0

        adjacency = self.get_adjacency_matrix()
        incidence = self.get_incidence_matrix()

        df = pd.DataFrame([{
            "Nodos": num_nodes,
            "Aristas": num_edges,
            "Partes conexas": num_components,
            "Grado máximo": grado_max,
            "Grado mínimo": grado_min,
            "Grado promedio": round(grado_prom, 2)
        }])

        return {
            "summary": df,
            "adjacency_matrix": adjacency,
            "incidence_matrix": incidence,
            "node_order": node_list
        }

    def __repr__(self) -> str:
        graph_type = "Directed" if self.directed else "Undirected"
        return (f"{self.__class__.__name__}({graph_type}, "
                f"Group={type(self.group).__name__}, "
                f"Nodes={self.graph.number_of_nodes()}, "
                f"Edges={self.graph.number_of_edges()})")
