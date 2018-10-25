import unittest
import numpy as np
from chariot.transformer.vocabulary import Vocabulary
from gcn.graph import DependencyGraph


class TestDependencyGraph(unittest.TestCase):

    def test_build(self):
        sentence = "I am living at house"
        vocab = Vocabulary()
        vocab.set(sentence.split())

        graph = DependencyGraph("en", vocab)

        sequence = vocab.transform([sentence.split()])[0]
        matrix = graph.build(sequence)

        answer = np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
        ])
        self.assertEqual(tuple(matrix.tolist()),
                         tuple(answer.tolist()))

    def test_batch_build(self):
        sentences = ["I am living at house",
                     "You are waiting on the station"]
        vocab = Vocabulary()
        vocab.set(sentences[0].split() + sentences[1].split())

        graph = DependencyGraph("en", vocab)
        sequences = vocab.transform(sentences)
        matrices = graph.batch_build(sequences, size=6)

        self.assertEqual(matrices.shape, (2, 6, 6))
