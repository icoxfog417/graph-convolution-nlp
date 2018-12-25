import unittest
import numpy as np
from chariot.transformer.vocabulary import Vocabulary
from gcn.graph import DependencyGraph


class TestDependencyGraph(unittest.TestCase):

    def test_build(self):
        graph = DependencyGraph("en")
        matrix = graph.build("I am living at house")

        answer = np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
        ])
        self.assertEqual(tuple(matrix.tolist()),
                         tuple(answer.tolist()))

    def test_build_label(self):
        graph = DependencyGraph("en")
        matrix = graph.build("I am living at house", return_label=True)

        answer = [
            ["", "", "nsubj", "", ""],
            ["", "", "aux", "", ""],
            ["", "", "ROOT", "", ""],
            ["", "", "prep", "", ""],
            ["", "", "", "pobj", ""],
        ]
        self.assertEqual(tuple(matrix),
                         tuple(answer))

    def test_batch_build(self):
        graph = DependencyGraph("en")

        sentences = ["I am living at house",
                     "You are waiting on the station"]
        matrices = graph.batch_build(sentences, size=6)

        self.assertEqual(matrices.shape, (2, 6, 6))
