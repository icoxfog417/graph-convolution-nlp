import os
import shutil
import unittest
from chariot.storage import Storage
from gcn.data.multinli_data import MultiNLIData


class TestData(unittest.TestCase):

    def test_data(self):
        root = os.path.join(os.path.dirname(__file__), "../")
        storage = Storage(root)

        r = MultiNLIData().download(storage.data_path("raw"))
        self.assertEqual(len(r.train_data), 392702)
        self.assertEqual(len(r.dev_data), 9815)
