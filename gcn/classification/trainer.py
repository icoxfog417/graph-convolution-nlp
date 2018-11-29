import numpy as np
import pandas as pd
from tensorflow.python import keras as K
import chariot.transformer as ct
from chariot.feeder import Feeder
from chariot.transformer.formatter import Padding
from gcn.base_trainer import BaseTrainer
from gcn.data.multi_nli_dataset import MultiNLIDataset


class Trainer(BaseTrainer):

    def download(self):
        r = MultiNLIDataset(self.storage.root).download()
        return r

    @property
    def num_classes(self):
        return len(MultiNLIDataset.labels())

    def build(self, data_kind="train", save=True):
        super().build(data_kind, "text", save)

    def train(self, model, data_kind="train", lr=1e-3,
              batch_size=20, sequence_length=25, epochs=40):
        if not self._built:
            raise Exception("Trainer's preprocessor is not built.")

        r = self.download()

        feeder = Feeder({"text": [
                            self.preprocessor,
                            Padding.from_(
                                self.preprocessor,
                                length=sequence_length)
                        ]})

        train_data = feeder.transform(r.train_data())
        test_data = feeder.transform(r.test_data())

        # Set optimizer
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=K.optimizers.Adam(lr=lr),
                      metrics=["accuracy"])

        model.fit(train_data["text"], train_data["label"],
                  validation_data=(test_data["text"], test_data["label"]),
                  batch_size=batch_size,
                  epochs=epochs)
