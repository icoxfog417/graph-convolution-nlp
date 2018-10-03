from tensorflow.python.keras import backend as K


def perplexity(y_true, y_pred):
    cross_entropy = K.mean(K.sparse_categorical_crossentropy(y_true, y_pred))
    perplexity = K.exp(cross_entropy)
    return perplexity
