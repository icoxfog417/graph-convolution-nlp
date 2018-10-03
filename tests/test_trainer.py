import os
import shutil
import unittest
from gcn.language_model.trainer import Trainer
from gcn.language_model.baseline import LSTMLM


class TestTrainer(unittest.TestCase):

    def test_download(self):
        root = os.path.join(os.path.dirname(__file__), "../")
        trainer = Trainer(root)

        r = trainer.download()
        self.assertTrue(r)

    def test_build(self):
        root = os.path.join(os.path.dirname(__file__), "../")
        trainer = Trainer(root, preprocessor_name="test_preprocessor")

        trainer.build("valid")
        self.assertTrue(len(trainer.preprocessor.vocabulary.get()) > 1000)
        print(trainer.preprocessor.vocabulary.get()[:100])
        print(trainer.preprocessor_path)
        os.remove(trainer.preprocessor_path)

    def test_train(self):
        root = os.path.join(os.path.dirname(__file__), "../")
        trainer = Trainer(root, preprocessor_name="test_train_preprocessor", log_dir="test")
        trainer.build(data_kind="valid")

        vocab_size = len(trainer.preprocessor.vocabulary.get())
        model = LSTMLM(vocab_size, embedding_size=100, hidden_size=50)

        metrics = trainer.train(model, data_kind="valid", epochs=2)
        last_acc = metrics.history["acc"][-1]
        shutil.rmtree(trainer.log_dir)
        os.remove(trainer.preprocessor_path)
        self.assertTrue(metrics.history["acc"][-1] - metrics.history["acc"][0] > 0)
