import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.layers import Dense


class SimpleAttentionLayer(Dense):

    def __init__(self,
                 feature_units,
                 activation="relu",
                 return_attention=False,
                 node_axis="row",
                 merge_method="add",
                 use_attention_kernel=True,
                 **kwargs):

        super(SimpleAttentionLayer, self).__init__(units=feature_units,
                                                   activation=activation,
                                                   **kwargs)
        if merge_method == "concat" and not use_attention_kernel:
            raise Exception("Can't use concat without attention")

        self.return_attention = return_attention
        self.node_axis = node_axis
        self.merge_method = merge_method
        self.use_attention_kernel = use_attention_kernel
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3)]
        self.supports_masking = False

        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        X_dims, A_dims = [dims.as_list() for dims in input_shape]
        assert len(X_dims) == 3
        assert len(A_dims) == 3 and A_dims[1] == A_dims[2]

        F = X_dims[-1]
        N = X_dims[1]

        self.kernel = self.add_weight(shape=(F, F),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      name="kernel")

        if self.use_bias:
            self.bias = self.add_weight(shape=(N, N),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        name="bias")

        self.built = True

    def call(self, inputs):
        X = inputs[0]  # Node features (B x N x F)
        A = inputs[1]  # Adjacency matrix (B x N x N)

        X_dims = X.get_shape().as_list()
        B, N, F = X_dims

        merged = tf.matmul(K.dot(X, self.self_kernel),
                           tf.transpose(X, (0, 2, 1)))
        attention = tf.nn.tanh(merged)
        attention = K.reshape(attention, (-1, N, N))

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
