from tensorflow.python import keras as K
from gcn.layers import ProjectionLayer
from gcn.util import gpu_enable


def LSTMLM(vocab_size, embedding_size=200, hidden_size=200,
           layers=1, dropout=0.2):
    # Prepare initializer
    initializer = K.initializers.RandomUniform(minval=-0.1, maxval=0.1)

    # Build the model
    model = K.Sequential()
    embedding = K.layers.Embedding(input_dim=vocab_size,
                                   output_dim=embedding_size,
                                   embeddings_initializer=initializer)
    model.add(embedding)
    model.add(K.layers.Dropout(dropout))
    rnn_layer = K.layers.CuDNNLSTM if gpu_enable() else K.layers.LSTM
    for layer in range(layers):
        return_sequences = True if layer < layers - 1 else False
        model.add(rnn_layer(hidden_size,
                            return_sequences=return_sequences,
                            recurrent_initializer="orthogonal"))
    model.add(K.layers.Dropout(dropout))
    model.add(K.layers.Dense(embedding_size, kernel_initializer=initializer))    
    model.add(ProjectionLayer(embedding))  # Tying encoder/decoder
    model.add(K.layers.Activation(activation="softmax"))

    return model
