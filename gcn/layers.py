import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import activations, constraints, initializers, regularizers
from tensorflow.python.keras.layers import Layer, Dropout, LeakyReLU


class ProjectionLayer(Layer):

    def __init__(self, embedding, **kwargs):
        super(ProjectionLayer, self).__init__(**kwargs)
        self.weight = embedding.embeddings
        self.output_dim = self.weight.shape[0]

    def call(self, x):
        return K.dot(x, K.transpose(self.weight))

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tf.TensorShape(output_shape)


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
        return input_shape[0],  self.embedding_dim


class GraphAttentionLayer(Layer):
    """
    import from danielegrattarola/keras-gat
    https://github.com/danielegrattarola/keras-gat/blob/master/keras_gat/graph_attention_layer.py
    """

    def __init__(self,
                 F_,
                 attn_heads=1,
                 attn_heads_reduction="concat",  # {"concat", "average"}
                 dropout_rate=0.5,
                 activation="relu",
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 attn_kernel_initializer="glorot_uniform",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):

        if attn_heads_reduction not in {"concat", "average"}:
            raise ValueError("Possbile reduction methods: concat, average")

        self.F_ = F_  # Number of output features (F" in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper
        self.dropout_rate = dropout_rate  # Internal dropout rate
        self.activation = activations.get(activation)  # Eq. 4 in the paper
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        # Populated by build()
        self.kernels = []       # Layer kernels for attention heads
        self.biases = []        # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads

        if attn_heads_reduction == "concat":
            # Output will have shape (..., K * F")
            self.output_dim = self.F_ * self.attn_heads
        else:
            # Output will have shape (..., F")
            self.output_dim = self.F_

        super(GraphAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        F = input_shape[0][-1]

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # Layer kernel
            kernel = self.add_weight(shape=(F, self.F_),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name="kernel_{}".format(head))
            self.kernels.append(kernel)

            # # Layer bias
            if self.use_bias:
                bias = self.add_weight(shape=(self.F_, ),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint,
                                       name="bias_{}".format(head))
                self.biases.append(bias)

            # Attention kernels
            attn_kernel_self = self.add_weight(shape=(self.F_, 1),
                                               initializer=self.attn_kernel_initializer,
                                               regularizer=self.attn_kernel_regularizer,
                                               constraint=self.attn_kernel_constraint,
                                               name="attn_kernel_self_{}".format(head),)
            attn_kernel_neighs = self.add_weight(shape=(self.F_, 1),
                                                 initializer=self.attn_kernel_initializer,
                                                 regularizer=self.attn_kernel_regularizer,
                                                 constraint=self.attn_kernel_constraint,
                                                 name="attn_kernel_neigh_{}".format(head))
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])
        self.built = True

    def call(self, inputs):
        X = inputs[0]  # Node features (N x F)
        A = inputs[1]  # Adjacency matrix (N x N)

        outputs = []
        for head in range(self.attn_heads):
            kernel = self.kernels[head]  # W in the paper (F x F")
            attention_kernel = self.attn_kernels[head]  # Attention kernel a in the paper (2F" x 1)

            # Compute inputs to attention network
            features = K.dot(X, kernel)  # (N x F")

            # Compute feature combinations
            # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
            attn_for_self = K.dot(features, attention_kernel[0])    # (N x 1), [a_1]^T [Wh_i]
            attn_for_neighs = K.dot(features, attention_kernel[1])  # (N x 1), [a_2]^T [Wh_j]

            # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
            dense = attn_for_self + K.transpose(attn_for_neighs)  # (N x N) via broadcasting

            # Add nonlinearty
            dense = LeakyReLU(alpha=0.2)(dense)

            # Mask values before activation (Vaswani et al., 2017)
            mask = -10e9 * (1.0 - A)
            dense += mask

            # Apply softmax to get attention coefficients
            dense = K.softmax(dense)  # (N x N)

            # Apply dropout to features and attention coefficients
            dropout_attn = Dropout(self.dropout_rate)(dense)  # (N x N)
            dropout_feat = Dropout(self.dropout_rate)(features)  # (N x F")

            # Linear combination with neighbors" features
            node_features = K.dot(dropout_attn, dropout_feat)  # (N x F")

            if self.use_bias:
                node_features = K.bias_add(node_features, self.biases[head])

            if self.attn_heads_reduction == "concat":
                # If "concat", compute the activation here (Eq. 5)
                node_features = self.activation(node_features)

            # Add output of attention head to final output
            outputs.append(node_features)

        # Aggregate the heads" output according to the reduction method
        if self.attn_heads_reduction == "concat":
            output = K.concatenate(outputs)  # (N x KF")
        else:
            output = K.mean(K.stack(outputs), axis=0)  # N x F")
            # If "average", compute the activation here (Eq. 6)
            output = self.activation(output)

        return output

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0][0], self.output_dim
        return output_shape
