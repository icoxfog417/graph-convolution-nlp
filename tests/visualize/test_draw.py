import unittest
import numpy as np
from chariot.transformer.vocabulary import Vocabulary
from gcn.graph import DependencyGraph
from gcn.visualize.draw import AttentionDrawer


class TestDraw(unittest.TestCase):

    def test_visualize_attention(self):
        sentence = "I am living at house"
        vocab = Vocabulary()
        vocab.set(sentence.split())

        graph_builder = DependencyGraph("en", vocab)
        sequence = vocab.transform([sentence.split()])[0]
        attention = np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 0.2, 0, 0],
            [0, 0, 0.7, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0.5, 0],
        ])

        drawer = AttentionDrawer(graph_builder)
        graph = drawer.draw(sequence, attention)
        drawer.show(graph)
