from tensorflow.python import keras as K
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import chazutsu
import chariot.transformer as ct
from chariot.feeder import LanguageModelFeeder
from gcn.base_trainer import BaseTrainer
from gcn.metrics import perplexity


class Trainer(BaseTrainer):

    def download(self):
        download_dir = self.storage.data_path("raw")
        matched = chazutsu.datasets.MultiNLI.matched().download(download_dir)
        mismatched = chazutsu.datasets.MultiNLI.mismatched().download(download_dir)
        return (matched, mismatched)

    @property
    def train_data(self):
        return self._merge_data("train")

    @property
    def test_data(self):
        return self._merge_data("test")

    def _merge_data(self, kind="train"):
        m, mm = self.download()
        datasets = []
        for d in [m, mm]:
            if kind == "train":
                _d = d.dev_data()
            else:
                _d = d.test_data()

            df = pd.concat([_d["sentence1"], _d["genre"]], axis=1)
            datasets.append(df)
        df = pd.concat(datasets).reset_index()
        df.rename(columns={"sentence1": "text", "genre": "label"}, inplace=True)
        return df

    def build(self, data_kind="train", save=True):
        if not self._built:
            self.load_preprocessor()
        if self._built:
            print("Load existing preprocessor at {}.".format(
                self.preprocessor_path))
            return 0

        data = self.train_data()["text"]
        print("Building Dictionary from {} data...".format(data_kind))
        self.preprocessor.fit(data)
        if save:
            joblib.dump(self.preprocessor, self.preprocessor_path)
        self._built = True
        print("Done!")

    def train(self, model, data_kind="train", lr=1e-3,
              batch_size=20, sequence_length=35, epochs=40):
        if not self.__built:
            raise Exception("Trainer's preprocessor is not built.")

        r = self.download()
        step_generators = {"train": {}, "valid": {}}

        # Set optimizer
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=K.optimizers.Adam(lr=lr),
                      metrics=["accuracy", perplexity])

        pass
