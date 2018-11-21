from tensorflow.python import keras as K
import numpy as np
import chazutsu
import chariot.transformer as ct
from chariot.feeder import LanguageModelFeeder
from gcn.base_trainer import BaseTrainer
from gcn.metrics import perplexity


class Trainer(BaseTrainer):

    def download(self):
        download_dir = self.storage.data_path("raw")
        r = chazutsu.datasets.WikiText2().download(download_dir)
        return r

    def train(self, model, data_kind="train", lr=1e-3,
              batch_size=20, sequence_length=35, epochs=40):
        if not self._built:
            raise Exception("Trainer's preprocessor is not built.")

        r = self.download()
        step_generators = {"train": {}, "valid": {}}

        # Set optimizer
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=K.optimizers.Adam(lr=lr),
                      metrics=["accuracy", perplexity])

        for k in step_generators:
            if k == "train":
                if data_kind == "train":
                    data = r.train_data()
                else:
                    data = r.valid_data()
            else:
                data = r.test_data()

            spec = {"sentence": ct.formatter.ShiftGenerator()}
            feeder = LanguageModelFeeder(spec)
            data = self.preprocessor.transform(data)
            step, generator = feeder.make_generator(
                                data, batch_size=batch_size,
                                sequence_length=sequence_length,
                                sequencial=False)

            step_generators[k]["g"] = generator
            step_generators[k]["s"] = step

        callbacks = [K.callbacks.ModelCheckpoint(self.model_path,
                                                 save_best_only=True),
                     K.callbacks.TensorBoard(self.tensorboard_dir)]

        metrics = model.fit_generator(
                    step_generators["train"]["g"](),
                    step_generators["train"]["s"],
                    validation_data=step_generators["valid"]["g"](),
                    validation_steps=step_generators["valid"]["s"],
                    epochs=epochs,
                    callbacks=callbacks)

        return metrics

    def generate_text(self, model, seed_text,
                      sequence_length=10, iteration=20):
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
            y = model.predict([x])
            index = min(len(preprocessed) - 1, sequence_length - 1)
            target_word_probs = y[index][0]
            w = np.random.choice(np.arange(len(target_word_probs)),
                                 1, p=target_word_probs)[0]
            preprocessed.append(w)

        decoded = self.preprocessor.inverse_transform([preprocessed])
        text = " ".join(decoded[0])

        return text
