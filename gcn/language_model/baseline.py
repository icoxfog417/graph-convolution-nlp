from tensorflow.python import keras as K


def LSTMLM(vocab_size, embedding_size, hidden_size):
    # Build model
    model = K.Sequential()
    embedding = K.layers.Embedding(input_dim=vocab_size, output_dim=embedding_size)
    model.add(embedding)
    model.add(K.layers.LSTM(hidden_size))
    model.add(K.layers.Dense(embedding_size))
    # Tying encoder/decoder
    model.add(K.layers.Lambda(lambda x: K.backend.dot(x, K.backend.transpose(
                                                      embedding.embeddings))))
    model.add(K.layers.Activation(activation="softmax"))

    # Set optimizer
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    return model
