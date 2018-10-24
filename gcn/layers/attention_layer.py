from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import initializers, regularizers, constraints


class AttentionLayer(Layer):
    """
    import from Bidirectional LSTM and Attention
    https://www.kaggle.com/takuok/bidirectional-lstm-and-attention-lb-0-043
    """

    def __init__(self, sequence_length,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, return_attentions=False, **kwargs):
        self.sequence_length = sequence_length
        self.supports_masking = True
        self.return_attentions = return_attentions
        self.init = initializers.get("glorot_uniform")

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.embedding_dim = 0
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        _input_shape = input_shape.as_list()
        self.embedding_dim = _input_shape[-1]
        self.W = self.add_weight(name="{}_W".format(self.name),
                                 shape=(self.embedding_dim,),
                                 initializer=self.init,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

        if self.bias:
            self.b = self.add_weight(name="{}_b".format(self.name),
                                     shape=(_input_shape[1],),
                                     initializer="zero",
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        embedding_dim = self.embedding_dim
        sequence_length = self.sequence_length

        eij = K.reshape(K.dot(K.reshape(x, (-1, embedding_dim)),
                              K.reshape(self.W, (embedding_dim, 1))),
                        (-1, sequence_length))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        weighted_input = x * K.expand_dims(a)
        output = K.sum(weighted_input, axis=1)
        if self.return_attentions:
            return output, a
        else:
            return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.embedding_dim
