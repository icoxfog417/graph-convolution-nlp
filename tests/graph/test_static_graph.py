import unittest
import numpy as np
from gcn.graph import StaticGraph


class TestStaticGraph(unittest.TestCase):

    def test_build(self):
        for k in ("self", "previous"):
            for f in (True, False):
                graph = StaticGraph("en", kind=k, fill=f)
                matrix = graph.build("You can get static graph.")
                self.check_graph(matrix, k, f)

    def check_graph(self, matrix, kind, fill):
        print("kind={}, fill={}".format(kind, fill))
        print(matrix)
        for r in range(len(matrix)):
            for c in range(len(matrix[r])):
                spike = False
                offset = 0 if kind == "self" else -1
                _r = r + offset
                if c == _r:
                    spike = True
                elif fill and c <= _r:
                    spike = True

                if spike:
                    self.assertEqual(matrix[r][c], 1)
                else:
                    self.assertEqual(matrix[r][c], 0)

    def test_batch_build(self):
        graph = StaticGraph("en")
        sentences = ["I am living at house",
                     "You are waiting on the station"]
        matrix = graph.batch_build(sentences, size=3)

        self.assertEqual(matrix.shape, (2, 3, 3))
