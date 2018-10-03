import os
import shutil
import unittest
from gcn.language_model.trainer import Trainer
from gcn.language_model.baseline import LSTMLM


class TestBaseline(unittest.TestCase):

    def test_lstm(self):
        root = os.path.join(os.path.dirname(__file__), "../")
        trainer = Trainer(root, preprocessor_name="test_baseline_preprocessor",
                          log_dir="test_baseline")
        trainer.build(data_kind="valid")

        vocab_size = len(trainer.preprocessor.vocabulary.get())
        model = LSTMLM(vocab_size, embedding_size=100, hidden_size=50)

        metrics = trainer.train(model, data_kind="valid", epochs=20)
        last_acc = metrics.history["acc"][-1]
        shutil.rmtree(trainer.log_dir)
        os.remove(trainer.preprocessor_path)
        self.assertTrue(metrics.history["acc"][-1] > 0.9)  # confirm overfit
