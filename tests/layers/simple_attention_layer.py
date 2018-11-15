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

        self.self_kernel = None
        self.neighbor_kernel = None
        self.attention_kernel = None
        self.bias = None

    def build(self, input_shape):
        X_dims, A_dims = [dims.as_list() for dims in input_shape]
        assert len(X_dims) == 3
        assert len(A_dims) == 3 and A_dims[1] == A_dims[2]

        F = X_dims[-1]
        N = X_dims[1]

        for kind in ["self", "neighbor", "attention"]:
            if kind in ["self", "neighbor"]:
                if self.use_attention_kernel:
                    shape = (F, self.units)
                else:
                    shape = (F, 1)
            elif kind == "attention" and self.use_attention_kernel:
                if self.merge_method == "concat":
                    shape = (self.units * 2, 1)
                else:
                    shape = (self.units, 1)
            else:
                shape = ()

            if len(shape) == 0:
                continue

            kernel = self.add_weight(shape=shape,
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name="{}_kernel".format(kind))

            if kind == "self":
                self.self_kernel = kernel
            elif kind == "neighbor":
                self.neighbor_kernel = kernel
            elif kind == "attention":
                self.attention_kernel = kernel

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

        feature_self = K.dot(X, self.self_kernel)
        feature_neighbor = K.dot(X, self.neighbor_kernel)

        # repeat_elements is same as np.repeat.
        # it repeats element to row direction.
        # Example.
        #  z = np.array([[1,2,3],[4,5,6]])  # shape=(2, 3)
        #  repeat = 4
        #  np.reshape(np.repeat(z, repeat, axis=-1), (2, 3, repeat))
        #  > array([[[1, 1, 1, 1],
        #          [2, 2, 2, 2],
        #          [3, 3, 3, 3]],
        #         [[4, 4, 4, 4],
        #          [5, 5, 5, 5],
        #          [6, 6, 6, 6]]])
        feature_self = K.repeat_elements(feature_self, N, axis=2)
        feature_self = K.reshape(feature_self, (-1, N, N, self.units))

        feature_neighbor = K.repeat_elements(feature_neighbor, N, axis=2)
        feature_neighbor = K.reshape(feature_neighbor, (-1, N, N, self.units))

        T = (0, 2, 1, 3)
        if self.merge_method == "concat":
            if self.node_axis == "row":
                merged = tf.concat([feature_self,
                                    tf.transpose(feature_neighbor, T)],
                                   axis=-1)
            else:
                merged = tf.concat([tf.transpose(feature_self, T),
                                    feature_neighbor],
                                   axis=-1)
        else:
            if self.node_axis == "row":
                merged = feature_self + tf.transpose(feature_neighbor, T)
            else:
                merged = tf.transpose(feature_self, T) + feature_neighbor

        activation_func = tf.nn.tanh
        if self.use_attention_kernel:
            attention = K.dot(activation_func(merged), self.attention_kernel)
        else:
            attention = activation_func(merged)

        """
        print([self.merge_method, self.node_axis, self.use_attention_kernel])
        print(merged.shape)
        print(attention.shape)
        """
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
