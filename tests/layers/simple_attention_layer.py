import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.layers import Dense


class SimpleAttentionLayer(Dense):

    def __init__(self,
                 feature_units,
                 activation="relu",
                 return_attention=False,
                 **kwargs):

        super(SimpleAttentionLayer, self).__init__(units=feature_units,
                                                   activation=activation,
                                                   **kwargs)

        self.return_attention = return_attention
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3)]
        self.supports_masking = False

        self.self_kernel = None
        self.neighbor_kernel = None
        self.attention_kernel = None
        self.bias = None

    def build(self, input_shape):
        X_dims, A_dims = [dims.as_list() for dims in input_shape]
        assert len(X_dims) == 3
        assert len(A_dims) == 3 and A_dims[1] == A_dims[2]

        F = X_dims[-1]

        for kind in ["self", "neighbor", "attention"]:
            if kind == "self":
                shape = (F, self.units)
            elif kind == "neighbor":
                shape = (F, self.units)
            else:
                shape = (self.units, 1)

            kernel = self.add_weight(shape=shape,
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name="{}_kernel".format(kind))

            if kind == "self":
                self.self_kernel = kernel
            elif kind == "neighbor":
                self.neighbor_kernel = kernel
            else:
                self.attention_kernel = kernel

        if self.use_bias:
            self.bias = self.add_weight(shape=shape[1],
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        name="bias")

        self.built = True

    def call(self, inputs):
        X = inputs[0]  # Node features (B x N x F)
        A = inputs[1]  # Adjacency matrix (B x N x N)

        X_dims = X.get_shape().as_list()
        batch_size, node_count, feature_size = X_dims

        feature_self = K.dot(X, self.self_kernel)
        feature_neighbor = K.dot(X, self.neighbor_kernel)

        feature_self = K.repeat_elements(feature_self, node_count, axis=2)
        feature_self = K.reshape(feature_self, (-1, node_count, node_count, self.units))

        feature_neighbor = K.repeat_elements(feature_neighbor, node_count, axis=2)
        feature_neighbor = K.reshape(feature_neighbor, (-1, node_count, node_count, self.units))

        additive = feature_self + tf.transpose(feature_neighbor, (0, 2, 1, 3))

        attention = K.dot(tf.nn.tanh(additive), self.attention_kernel)
        attention = K.reshape(attention, (-1, node_count, node_count))

        if self.use_bias:
            attention = K.bias_add(attention, self.bias)

        mask = -10e9 * (1.0 - A)
        attention += mask

        attention = tf.nn.softmax(attention)

        output = tf.matmul(attention, X)

        if self.return_attention:
            return (output, attention)
        else:
            return output

    def compute_output_shape(self, input_shape):
        X_dims, A_dims = [dims.as_list() for dims in input_shape]
        assert len(X_dims) == 3
        assert len(A_dims) == 3
        output_shape = X_dims[0], X_dims[0], self.output_dim

        if self.return_attention:
            return (tf.TensorShape(output_shape),
                    tf.TensorShape(A_dims))
        else:
            return tf.TensorShape(output_shape)
