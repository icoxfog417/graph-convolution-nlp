import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class AttentionDrawer():

    def __init__(self, graph_builder):
        self.graph_builder = graph_builder

    def draw(self, sequence, attention):
        vocabulary = self.graph_builder.vocabulary
        words = vocabulary.inverse(sequence)
        graph = self._build(words, attention)
        return graph

    def _build(self, nodes, matrix):
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        for i, row in enumerate(matrix):
            for j, col in enumerate(row):
                if matrix[i][j] > 0:
                    graph.add_edge(nodes[i], nodes[j],
                                   weight=matrix[i][j])

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
        plt.axis("off")
        plt.tight_layout()
        plt.show()
