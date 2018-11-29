import os
import unittest
from gcn.data.multi_nli_dataset import MultiNLIDataset
from gcn.classification.baseline import TfidfClassifier


class TestBaseline(unittest.TestCase):

    def test_baseline(self):
        root = os.path.join(os.path.dirname(__file__), "../../")
        dataset = MultiNLIDataset(root)
        data = dataset.test_data()

        classifier = TfidfClassifier()
        scores = classifier.fit(data["text"], data["label"])
        self.assertTrue(len(scores) > 0)
