import unittest
from gcn.language_model.similarity_graph_lm import SimilarityGraphLM


class TestSimilarityGraphLM(unittest.TestCase):

    def test_similarity_graph_lm(self):
        vocab_size = 100
        sequence_length = 15
        embedding_size = 10
        model = SimilarityGraphLM(vocab_size, sequence_length,
                                  embedding_size)
