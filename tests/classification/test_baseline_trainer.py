import os
import unittest
from gcn.classification.baseline_trainer import BaselineTrainer
from gcn.classification.baseline import LSTMClassifier


class TestBaselineTrainer(unittest.TestCase):

    def test_build(self):
        root = os.path.join(os.path.dirname(__file__), "../../")
        trainer = BaselineTrainer(root, preprocessor_name="test_cbt_preprocessor")

        trainer.build()
        self.assertTrue(len(trainer.preprocessor.vocabulary.get()) > 1000)
        print(trainer.preprocessor.vocabulary.get()[:100])
        print(trainer.preprocessor_path)
        os.remove(trainer.preprocessor_path)

    def test_train(self):
        root = os.path.join(os.path.dirname(__file__), "../../")
        trainer = BaselineTrainer(root, preprocessor_name="test_cbt_preprocessor")
        trainer.build()

        vocab_size = len(trainer.preprocessor.vocabulary.get())
        model = LSTMClassifier(vocab_size)
        model.build(trainer.num_classes)

        metrics = trainer.train(model.model, epochs=2)
        self.assertTrue(metrics.history["acc"][-1] - metrics.history["acc"][0] > 0)
