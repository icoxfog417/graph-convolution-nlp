import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from gcn.graph.dependency_graph import DependencyGraph


class AttentionDrawer():

    def __init__(self, graph_builder):
        self.graph_builder = graph_builder

    def draw(self, sentence, attention=()):
        edge_matrix = ()
        nodes = self.graph_builder.get_nodes(sentence)

        size = len(attention) if len(attention) > 0 else len(nodes)
        if isinstance(self.graph_builder, DependencyGraph):
            edge_matrix = self.graph_builder.build(
                            sentence, size, return_label=True)
        matrix = attention
        if len(attention) == 0:
            matrix = self.graph_builder.build(sentence, size)
        graph = self._build(nodes, matrix, edge_matrix)
        return graph

    def _build(self, nodes, matrix, edge_matrix=()):
        graph = nx.Graph()
        graph.add_nodes_from(nodes[i] for i in range(len(matrix)))
        print(matrix)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
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
        width = 1 + (np.abs(weights) * max_width - 1)

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
