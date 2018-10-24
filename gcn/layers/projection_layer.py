import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer


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
