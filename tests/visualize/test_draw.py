import unittest
import numpy as np
from chariot.transformer.vocabulary import Vocabulary
from gcn.graph import DependencyGraph, SimilarityGraph, StaticGraph
from gcn.visualize.draw import AttentionDrawer


class TestDraw(unittest.TestCase):

    def test_draw_dependency_graph(self):
        sentence = "I am living at house"
        graph_builder = DependencyGraph("en")
        attention = np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 0.2, 0, 0],
            [0, 0, 0.7, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0.5, 0],
        ])

        drawer = AttentionDrawer(graph_builder)
        graph = drawer.draw(sentence, attention)
        drawer.show(graph)

    def test_draw_similarity_graph(self):
        sentence = "I am building similarity graph structure"
        graph_builder = SimilarityGraph("en")
        drawer = AttentionDrawer(graph_builder)
        graph = drawer.draw(sentence)
        drawer.show(graph)

    def test_draw_static_graph(self):
        sentence = "I am static graph"
        graph_builder = StaticGraph("en", kind="previous")
        drawer = AttentionDrawer(graph_builder)
        graph = drawer.draw(sentence)
        drawer.show(graph)
