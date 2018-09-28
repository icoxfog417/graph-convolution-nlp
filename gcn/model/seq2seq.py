import os
import numpy as np
from tensorflow.python import keras as K
from src.utils.storage import Storage


class SimpleSeq2Seq():

    def __init__(self, vocab_size, embedding_size, hidden_size):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.model = None
        self._predict_model = ()

    def build(self):
        # Encoder with embedding
        encoder_inputs = K.layers.Input(shape=(None,))
        embedding = K.layers.Embedding(self.vocab_size, self.embedding_size)
        encoder = K.layers.LSTM(self.hidden_size, return_state=True)

        embed_seq = embedding(encoder_inputs)
        _, hidden, context = encoder(embed_seq)
        encoder_state = [hidden, context]

        # Decoder
        decoder_inputs = K.layers.Input(shape=(None,))
        decoder = K.layers.LSTM(self.hidden_size, return_state=True,
                                return_sequences=True)
        decoded_seq, _, _ = decoder(embedding(decoder_inputs),
                                    initial_state=encoder_state)

        # Projection (use tying to save weight and regularization)
        proj = K.layers.Lambda(lambda x: K.backend.dot(x, K.backend.transpose(
                                                       embedding.embeddings)))
        projected = K.layers.Activation("softmax")(proj(decoded_seq))

        # Set Model
        self.model = K.models.Model(inputs=[encoder_inputs, decoder_inputs],
                                    outputs=projected)

        # Make predict Model
        predict_encoder = K.models.Model(encoder_inputs, encoder_state)
        decoder_state = [K.layers.Input(shape=(self.hidden_size,)),
                         K.layers.Input(shape=(self.hidden_size,))]
        d_out, d_h, d_c = decoder(embedding(decoder_inputs),
                                  initial_state=decoder_state)
        output = K.layers.Activation("softmax")(proj(d_out))
        next_state = [d_h, d_c]
        predict_decoder = K.models.Model([decoder_inputs] + decoder_state,
                                         [output] + next_state)

        self._predict_model = (predict_encoder, predict_decoder)

    def load(self, model_path):
        self.model.load_weights(model_path)

    def inference(self, input_seq, start_index, stop_index, max_size=50):
        predict_encoder, predict_decoder = self._predict_model

        # Make encoded states
        decoder_state = predict_encoder.predict(input_seq)

        # Make target seq
        target = [start_index]

        stop = False
        decoded = []
        while not stop:
            _target = np.array([target])
            output_tokens, h, c = predict_decoder.predict(
                                    [_target] + decoder_state)
            selected_token = np.random.choice(list(range(self.vocab_size)),
                                              size=1,
                                              p=output_tokens[0, -1, :])[0]
            decoded.append(selected_token)

            if selected_token == stop_index:
                stop = True
            elif len(decoded) > max_size:
                stop = True

            target.append(selected_token)

        decoded = [t for t in decoded if t not in (start_index, stop_index)]
        return decoded


class Seq2SeqTrainer():

    def __init__(self, model):
        self.model = model

    def train(self, seq2seq_dataset, batch_size=32,
              epochs=15, x_padding=-1, y_padding=-1,
              optimizer="rmsprop"):

        storage = Storage()
        log_dir, model_name = storage.make_logdir(self.model)
        X, y = seq2seq_dataset.get_Xy_padded(x_padding, y_padding)
        self.model.model.compile(optimizer=optimizer,
                                 loss="categorical_crossentropy")
        model_file = os.path.join(log_dir, "{}.h5".format(model_name))
        callbacks = [K.callbacks.TensorBoard(log_dir=log_dir),
                     K.callbacks.ModelCheckpoint(model_file,
                                                 save_best_only=True,
                                                 save_weights_only=True)]

        # Begin training
        y, y_teacher = self.make_label(y)
        y_teacher = K.utils.to_categorical(y_teacher, self.model.vocab_size)
        self.model.model.fit([X, y], y_teacher,
                             validation_split=0.2,
                             batch_size=batch_size, epochs=epochs,
                             callbacks=callbacks)
        return log_dir

    def make_label(self, y):
        y_teacher = np.roll(y, -1, axis=1)
        y = y[:, :-1]  # exclude end
        y_teacher = y_teacher[:, :-1]
        return y, y_teacher
