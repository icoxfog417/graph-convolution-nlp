from tensorflow.python import keras as K
from gcn.layers import ProjectionLayer
from gcn.util import gpu_enable


def LSTMLM(vocab_size, embedding_size=100, hidden_size=100,
           layers=1, dropout=0.5):
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
        model.add(rnn_layer(hidden_size, return_sequences=True))
    model.add(K.layers.Dropout(dropout))
    if hidden_size != embedding_size:
        model.add(K.layers.TimeDistributed(
                    K.layers.Dense(embedding_size,
                                   kernel_initializer=initializer)
                ))
    # Tying encoder/decoder
    #model.add(K.layers.TimeDistributed(ProjectionLayer(embedding)))
    model.add(K.layers.TimeDistributed(
            K.layers.Dense(vocab_size,
                           kernel_initializer=initializer, activation="softmax")
        ))

    #model.add(K.layers.Activation(activation="softmax"))

    return model
