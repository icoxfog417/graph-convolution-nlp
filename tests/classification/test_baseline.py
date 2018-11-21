import os
import unittest
from gcn.classification.trainer import Trainer
from gcn.classification.baseline import TfidfClassifier


class TestBaseline(unittest.TestCase):

    def test_baseline(self):
        root = os.path.join(os.path.dirname(__file__), "../../")
        trainer = Trainer(root)
        data = trainer.train_data

        classifier = TfidfClassifier()
        scores = classifier.fit(data["text"], data["label"])
        self.assertTrue(len(scores) > 0)
