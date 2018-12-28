import os
import shutil
import unittest
import numpy as np
from gcn.classification.trainer import Trainer
from gcn.classification.graph_based_classifier import GraphBasedClassifier
from gcn.data.multi_nli_dataset import MultiNLIDataset
from gcn.graph.dependency_graph import DependencyGraph
from gcn.graph.similarity_graph import SimilarityGraph
from gcn.graph.static_graph import StaticGraph


class TestTrainer(unittest.TestCase):

    def test_train_by_dependency_graph(self):
        self._test_train("dependency")

    def test_train_by_similarity_graph(self):
        self._test_train("similarity")

    def test_train_by_static_graph(self):
        self._test_train("static")

    def _test_train(self, graph_type):
        root = os.path.join(os.path.dirname(__file__), "../../")
        sequence_length = 25
        heads = 3

        dataset = MultiNLIDataset(root)
        test_data = dataset.test_data()
        index = np.random.randint(len(test_data), size=1)[0]
        text = test_data["text"].iloc[index]

        graph_builder = None
        if graph_type == "dependency":
            graph_builder = DependencyGraph(lang="en")
        elif graph_type == "similarity":
            graph_builder = SimilarityGraph(lang="en")
        else:
            graph_builder = StaticGraph(lang="en")

        trainer = Trainer(graph_builder, root,
                          preprocessor_name="test_ct_preprocessor")

        trainer.build(data_kind="test")

        def preprocessor(x):
            _x = trainer.preprocess(x, sequence_length)
            values = (_x["text"], _x["graph"])
            return values

        _, g = preprocessor([text])
        vocab_size = len(trainer.preprocessor.vocabulary.get())
        model = GraphBasedClassifier(vocab_size, sequence_length, heads=heads)
        model.build(trainer.num_classes, preprocessor)

        metrics = trainer.train(model.model, epochs=2)
        os.remove(trainer.preprocessor_path)
        self.assertTrue(metrics.history["acc"][-1] - metrics.history["acc"][0] > 0)

        attention = model.show_attention([text])
        self.assertEqual(len(attention), 1)  # batch size
        attention = attention[0]
        self.assertEqual(len(attention), 2)  # layer count
        attention = attention[0]
        self.assertEqual(attention.shape, (heads, sequence_length, sequence_length))
