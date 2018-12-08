import os
import shutil
import unittest
from gcn.classification.trainer import Trainer
from gcn.classification.graph_based_classifier import GraphBasedClassifier
from gcn.graph.dependency_graph import DependencyGraph
from gcn.graph.similarity_graph import SimilarityGraph


class TestTrainer(unittest.TestCase):

    def test_build_dependency_graph_trainer(self):
        self._test_build("dependency")

    def test_build_similarity_graph_trainer(self):
        self._test_build("similarity")

    def _test_build(self, graph_type):
        root = os.path.join(os.path.dirname(__file__), "../../")
        trainer = Trainer(root, preprocessor_name="test_ct_preprocessor")

        if graph_type == "dependency":
            trainer.build_dependency_graph_trainer()
        else:
            trainer.build_similarity_graph_trainer()

        self.assertTrue(len(trainer.preprocessor.vocabulary.get()) > 1000)
        if graph_type == "dependency":
            self.assertTrue(isinstance(trainer.graph_builder, DependencyGraph))
        else:
            self.assertTrue(isinstance(trainer.graph_builder, SimilarityGraph))
        os.remove(trainer.preprocessor_path)

    def test_train_by_dependency_graph(self):
        self._test_train("dependency")

    def test_train_by_similarity_graph(self):
        self._test_train("similarity")

    def _test_train(self, graph_type):
        root = os.path.join(os.path.dirname(__file__), "../../")
        trainer = Trainer(root, preprocessor_name="test_ct_preprocessor")

        if graph_type == "dependency":
            trainer.build_dependency_graph_trainer()
        else:
            trainer.build_similarity_graph_trainer()

        vocab_size = len(trainer.preprocessor.vocabulary.get())
        sequence_length = 25
        model = GraphBasedClassifier(vocab_size, sequence_length)
        model.build(trainer.num_classes)

        metrics = trainer.train(model.model, epochs=2)
        self.assertTrue(metrics.history["acc"][-1] - metrics.history["acc"][0] > 0)
