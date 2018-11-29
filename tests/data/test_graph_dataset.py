import os
import unittest
from gcn.data.graph_dataset import GraphDataset


class TestGraphDataset(unittest.TestCase):

    def test_citeseer(self):
        root = os.path.join(os.path.dirname(__file__), "../../")
        gd = GraphDataset(root, kind="citeseer")
        x, y, tx, ty, allx, ally, graph, test_idx = gd.download()
