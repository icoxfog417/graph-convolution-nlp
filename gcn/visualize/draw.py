import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from gcn.graph.dependency_graph import DependencyGraph


class AttentionDrawer():

    def __init__(self, graph_builder):
        self.graph_builder = graph_builder

    def draw(self, sequence, attention):
        vocabulary = self.graph_builder.vocabulary
        words = vocabulary.inverse(sequence)
        edge_matrix = ()
        if isinstance(self.graph_builder, DependencyGraph):
            size = len(attention)
            edge_matrix = self.graph_builder.build(
                            sequence, size, return_label=True)
        graph = self._build(words, attention, edge_matrix)
        return graph

    def _build(self, nodes, matrix, edge_matrix=()):
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        for i, row in enumerate(matrix):
            for j, col in enumerate(row):
                if matrix[i][j] > 0:
                    if len(edge_matrix) == 0:
                        graph.add_edge(nodes[i], nodes[j],
                                       weight=matrix[i][j])
                    else:
                        graph.add_edge(nodes[i], nodes[j],
                                       weight=matrix[i][j],
                                       label=edge_matrix[i][j])

        return graph

    def show(self, graph, figsize=(6, 6),
             node_color="skyblue", edge_color="grey",
             font_size=15, max_width=5):
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(graph)
        weights = np.array([graph[u][v]["weight"] for u, v in graph.edges()])
        width = 1 + (weights * max_width - 1)

        nx.draw_networkx(graph, pos,
                         node_color=node_color,
                         font_size=font_size, edge_color=edge_color,
                         width=width)

        if isinstance(self.graph_builder, DependencyGraph):
            labels = {(u, v): graph[u][v]["label"] for u, v in graph.edges()}
            nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)

        plt.axis("off")
        plt.tight_layout()
        plt.show()
