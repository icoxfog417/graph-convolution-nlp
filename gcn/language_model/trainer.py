import os
import sys
import numpy as np
from sklearn.externals import joblib
from tensorflow.python import keras as K
from chariot.storage import Storage
import chariot.transformer as ct
from chariot.preprocessor import Preprocessor
from chariot.feeder import LanguageModelFeeder
from gcn.data.multinli_data import MultiNLIData


class Trainer():

    def __init__(self, root="", lang=None, min_df=5, max_df=sys.maxsize,
                 unknown="<unk>", train_data_limit=-1,
                 preprocessor_name="preprocessor", log_dir=""):
        default_root = os.path.join(os.path.dirname(__file__), "../../")
        _root = root if root else default_root

        self.storage = Storage(_root)
        self.preprocessor_name = preprocessor_name
        self.train_data_limit = train_data_limit
        self.__log_dir = log_dir
        self.__built = False
        self.preprocessor = Preprocessor(
                                text_transformers=[
                                    ct.text.UnicodeNormalizer(),
                                    ct.text.LowerNormalizer()
                                ],
                                tokenizer=ct.Tokenizer(lang=lang),
                                vocabulary=ct.Vocabulary(
                                            min_df=min_df, max_df=max_df,
                                            unknown=unknown))

        if os.path.exists(self.preprocessor_path):
            self.__built = True
            self.preprocessor = joblib.load(self.preprocessor_path)

    @property
    def preprocessor_path(self):
        path = "interim/{}.pkl".format(self.preprocessor_name)
        return self.storage.data_path(path)

    @property
    def _log_dir(self):
        folder = "/" + self.__log_dir if self.__log_dir else ""
        log_dir = "log{}".format(folder)
        if not os.path.exists(self.storage.data_path(log_dir)):
            os.mkdir(self.storage.data_path(log_dir))

        return log_dir

    @property
    def log_dir(self):
        return self.storage.data_path(self._log_dir)

    @property
    def model_path(self):
        return self.storage.data_path(self._log_dir + "/model.h5")

    @property
    def tensorboard_dir(self):
        return self.storage.data_path(self._log_dir)

    def download(self):
        download_dir = self.storage.data_path("raw")
        r = MultiNLIData().download(download_dir)
        return r

    def _limited_train(self, r):
        train_data = r.train_data
        if self.train_data_limit > 0:
            train_data = train_data[:self.train_data_limit]
        return train_data

    def build(self, data_kind="train", save=True):
        r = self.download()
        data = self._limited_train(r) if data_kind == "train" else r.dev_data
        print("Building Dictionary from {} data...".format(data_kind))
        self.preprocessor.fit(data["sentence"])
        if save:
            joblib.dump(self.preprocessor, self.preprocessor_path)
        self.__built = True
        print("Done!")

    def train(self, model, batch_size=32, sequence_length=8, epochs=50):
        if not self.__built:
            raise Exception("Trainer's preprocessor is not trained.")

        r = self.download()
        step_generators = {"train": {}, "dev": {}}

        for k in step_generators:
            if k == "train":
                data = self._limited_train(r)
            else:
                data = r.dev_data

            spec = {"sentence": ct.formatter.ShiftGenerator()}
            feeder = LanguageModelFeeder(spec)
            data = self.preprocessor.transform(data)
            step, generator = feeder.make_generator(
                                data, batch_size=batch_size,
                                sequence_length=sequence_length,
                                sequencial=False)

            step_generators[k]["g"] = generator
            step_generators[k]["s"] = step

        callbacks = [K.callbacks.ModelCheckpoint(self.model_path, save_best_only=True),
                     K.callbacks.TensorBoard(self.tensorboard_dir)]

        metrics = model.fit_generator(
                    step_generators["train"]["g"](),
                    step_generators["train"]["s"],
                    validation_data=step_generators["dev"]["g"](),
                    validation_steps=step_generators["dev"]["s"],
                    epochs=epochs,
                    callbacks=callbacks)

        return metrics

    def generate_text(self, seed_text, model, sequence_length=10, iteration=20):
        preprocessed = self.preprocessor.transform([seed_text])[0]

        def pad_sequence(tokens, length):
            if len(tokens) < length:
                pad_size = length - len(tokens)
                return tokens + [self.preprocessor.vocabulary.pad] * pad_size
            elif len(tokens) > length:
                return tokens[-length:]
            else:
                return tokens

        for _ in range(iteration):
            x = pad_sequence(preprocessed, sequence_length)
            y = model.predict([x])[0]
            w = np.random.choice(np.arange(len(y)), 1, p=y)[0]
            preprocessed.append(w)

        decoded = self.preprocessor.inverse_transform([preprocessed])
        text = " ".join(decoded[0])

        return text
