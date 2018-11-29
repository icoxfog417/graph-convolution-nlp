import os
import unittest
from gcn.data.multi_nli_dataset import MultiNLIDataset


class TestMultiNLIDataset(unittest.TestCase):

    def test_download(self):
        root = os.path.join(os.path.dirname(__file__), "../../")
        dataset = MultiNLIDataset(root, prefix="test")
        dataset.download()

        train_data = dataset.train_data()
        test_data = dataset.test_data()

        for d in [train_data, test_data]:
            self.assertTrue(len(d) > 0)
            counts = d["label"].value_counts().values.tolist()
            c = counts[0]
            for _c in counts:
                self.assertEqual(c, _c)

        for k in ["train", "test"]:
            self.assertTrue(os.path.exists(dataset.interim_file(k)))
            os.remove(dataset.interim_file(k))

            self.assertTrue(os.path.exists(dataset.processed_file(k)))
            os.remove(dataset.processed_file(k))
