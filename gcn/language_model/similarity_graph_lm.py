import tensorflow as tf
from tensorflow.python import keras as K
from gcn.layers import GraphAttentionLayer
from gcn.metrics import perplexity


def SimilarityGraphLM(vocab_size, sequence_length,
                      embedding_size, dropout=0.7, num_graph_conv=2):

    words = K.layers.Input(shape=(sequence_length,))
    matrix = K.layers.Input(shape=(sequence_length,))

    embeddings = K.layers.Embedding(output_dim=embedding_size,
                                    input_dim=vocab_size,
                                    input_length=sequence_length)(words)

    # context feature
    context = K.layers.LSTM(embedding_size, dropout=dropout,
                            return_sequences=True, return_state=True)(embeddings)

    # graph feature
    features = tf.transpose(embeddings, [1, 0, 2])
    for layer in range(num_graph_conv):
        features = K.layers.TimeDistributed(
                        GraphAttentionLayer(
                            embedding_size,
                            attn_heads_reduction="average"))([features, matrix])
    return None

    features = K.backend.transpose(features)
    merged = K.layers.concatenate([context, features])
    output = K.layers.Dense(vocab_size, activation="softmax")(merged)

    model = K.models.Model(inputs=[words, matrix], outputs=output)

    # Set optimizer
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy", perplexity])

    return model
