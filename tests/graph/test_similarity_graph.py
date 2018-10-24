import os
import unittest
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from chariot.transformer.vocabulary import Vocabulary
from gcn.graph import SimilarityGraph


class TestSimilarityGraph(unittest.TestCase):

    def test_build(self):
        root = os.path.join(os.path.dirname(__file__), "../../")
        graph = SimilarityGraph(root, graph_name="test_similarity_graph")

        nearest_neighbor = 3
        node_count = 10
        feature_size = 5

        vectors = np.random.uniform(size=node_count * feature_size)
        vectors = vectors.reshape(node_count, feature_size)

        similarity = cosine_similarity(vectors)
        similarity -= np.eye(node_count)
        top_k = np.argsort(-similarity, axis=1)[:, :nearest_neighbor]

        for mode in ["connectivity", "distance"]:
            matrix = graph._build(vectors, nearest_neighbor=nearest_neighbor,
                                  mode=mode)

            for i, top in enumerate(top_k):
                if mode == "connectivity":
                    self.assertEqual(sum(matrix[i, top]), nearest_neighbor)
                else:
                    self.assertEqual(tuple(similarity[i, top]),
                                     tuple(matrix[i, top]))

    def test_build_from_vocab(self):
        root = os.path.join(os.path.dirname(__file__), "../../")
        graph = SimilarityGraph(root, graph_name="test_similarity_graph")

        vocab = Vocabulary()
        vocab.set(["you", "and", "I", "loaded", "word", "vector", "now"])

        matrix = graph.build(vocab, nearest_neighbor=2)
        os.remove(graph.path)
