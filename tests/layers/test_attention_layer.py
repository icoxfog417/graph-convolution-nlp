import unittest
import numpy as np
from tensorflow.python import keras as K
from gcn.layers import AttentionLayer


class TestAttentionLayer(unittest.TestCase):

    def test_attention_layer(self):
        sample_size = 20000
        sequence_length = 20
        embedding_size = 2
        units = 8
        attention_column = 3

        # Baseline Model
        base = K.Sequential()
        base.add(K.layers.LSTM(units,
                               input_shape=[sequence_length, embedding_size]))
        base.add(K.layers.Dense(1, activation="sigmoid"))
        base.compile(optimizer="adam", loss="binary_crossentropy",
                     metrics=["accuracy"])

        # Attention Model
        def make_model():
            input = K.layers.Input([sequence_length, embedding_size])
            lstm_out = K.layers.LSTM(units, return_sequences=True)(input)
            attn_out, prob = AttentionLayer(sequence_length,
                                            return_attentions=True)(lstm_out)
            output = K.layers.Dense(1, activation="sigmoid")(attn_out)
            model = K.models.Model(inputs=input, outputs=output)
            return model

        model = make_model()
        model.compile(optimizer="adam", loss="binary_crossentropy",
                      metrics=["accuracy"])
        x, y = self.make_test_data(sample_size, sequence_length,
                                   embedding_size, attention_column)

        base_metrics = base.fit(x, y, epochs=1, batch_size=32,
                                validation_split=0.1, verbose=1)
        metrics = model.fit(x, y, epochs=1, batch_size=32,
                            validation_split=0.1, verbose=1)

        base_score = base_metrics.history["val_acc"][-1]
        score = metrics.history["val_acc"][-1]
        self.assertTrue(score > base_score)

        attention_layer = model.layers[2]
        attention_model = K.models.Model(inputs=model.input,
                                         outputs=attention_layer.output)
        activation, attention = attention_model.predict_on_batch(x)
        attention_index = np.argmax(np.mean(attention, axis=0))
        print(np.mean(attention, axis=0))
        self.assertTrue(attention_index in [attention_column,
                                            attention_column + 1])

    def make_test_data(self, sample_size, sequence_length, embedding_size,
                       attention_column):
        if attention_column >= sequence_length:
            raise Exception("Directed column is larger than sequence_length.")

        x = np.random.standard_normal(size=(sample_size, sequence_length,
                                            embedding_size))
        y = np.random.randint(low=0, high=2, size=(sample_size, 1))
        x[:, attention_column, :] = np.tile(y[:], (1, embedding_size))

        return x, y
