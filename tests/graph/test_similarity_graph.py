import os
import unittest
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from chariot.transformer.vocabulary import Vocabulary
from gcn.graph import SimilarityGraph


class TestSimilarityGraph(unittest.TestCase):

    def test_build(self):
        root = os.path.join(os.path.dirname(__file__), "../../")
        nearest_neighbor = 3
        node_count = 10
        feature_size = 5

        graph = SimilarityGraph("en", nearest_neighbor, root=root)

        vectors = np.random.uniform(size=node_count * feature_size)
        vectors = vectors.reshape(node_count, feature_size)

        similarity = cosine_similarity(vectors)
        similarity -= np.eye(node_count)
        top_k = np.argsort(-similarity, axis=1)[:, :nearest_neighbor]

        for mode in ["connectivity", "distance"]:
            graph.mode = mode
            matrix = graph._build(vectors)

            for i, top in enumerate(top_k):
                if mode == "connectivity":
                    self.assertEqual(sum(matrix[i, top]), nearest_neighbor)
                else:
                    self.assertEqual(tuple(similarity[i, top]),
                                     tuple(matrix[i, top]))

    def test_build_from_vocab(self):
        root = os.path.join(os.path.dirname(__file__), "../../")
        graph = SimilarityGraph("en", nearest_neighbor=2, root=root)
        matrix = graph.build("you loaded now")
        self.assertTrue(matrix.shape, (3, 3))

    def test_batch_build(self):
        root = os.path.join(os.path.dirname(__file__), "../../")
        sentences = ["I am living at house",
                     "You are waiting on the station"]
        graph = SimilarityGraph("en", nearest_neighbor=2, root=root)
        matrices = graph.batch_build(sentences, size=6)

        self.assertEqual(matrices.shape, (2, 6, 6))
