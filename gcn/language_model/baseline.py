from tensorflow.python import keras as K
from gcn.language_model.metrics import perplexity


def LSTMLM(vocab_size, embedding_size, hidden_size, dropout=0.7):
    # Build model
    model = K.Sequential()
    embedding = K.layers.Embedding(input_dim=vocab_size, output_dim=embedding_size)
    model.add(embedding)
    model.add(K.layers.LSTM(hidden_size, dropout=dropout))
    model.add(K.layers.Dense(embedding_size))
    # Tying encoder/decoder
    model.add(ProjectionLayer(embedding))
    model.add(K.layers.Activation(activation="softmax"))

    # Set optimizer
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy", perplexity])

    return model


class ProjectionLayer(K.layers.Layer):

    def __init__(self, embedding, **kwargs):
        super(ProjectionLayer, self).__init__(**kwargs)
        self.weight = embedding.embeddings
    
    def call(self, x):
        return K.backend.dot(x, K.backend.transpose(self.weight))
