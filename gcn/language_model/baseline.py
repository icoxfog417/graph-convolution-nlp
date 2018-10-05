from tensorflow.python import keras as K
from gcn.layers import ProjectionLayer
from gcn.metrics import perplexity


def LSTMLM(vocab_size, embedding_size, hidden_size,
           dropout=0.7, input_dropout=0.3):
    # Build model
    model = K.Sequential()
    embedding = K.layers.Embedding(input_dim=vocab_size,
                                   output_dim=embedding_size)
    model.add(embedding)
    model.add(K.layers.SpatialDropout1D(input_dropout))
    model.add(K.layers.LSTM(hidden_size, recurrent_dropout=dropout))
    model.add(K.layers.Dense(embedding_size))
    # Tying encoder/decoder
    model.add(ProjectionLayer(embedding))
    model.add(K.layers.Activation(activation="softmax"))

    # Set optimizer
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=K.optimizers.SGD(lr=1.0, clipnorm=5.0),
                  metrics=["accuracy", perplexity])

    return model
