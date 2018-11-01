import unittest
import numpy as np
from tensorflow.python import keras as K
from gcn.layers.graph_attention_layer import GraphAttentionLayer


class TestGraphAttentionLayer(unittest.TestCase):

    def test_forward(self):
        node_count = 12
        feature_size = 10
        feature_units = 8

        nodes = K.layers.Input(shape=(node_count, feature_size))
        matrix = K.layers.Input(shape=(node_count, node_count))
        layer = GraphAttentionLayer(feature_units=feature_units)
        output = layer([nodes, matrix])

        model = K.models.Model(inputs=[nodes, matrix], outputs=output)

        batch_size = 32
        node_samples = batch_size * node_count * feature_size
        node_inputs = np.random.uniform(size=node_samples)
        node_inputs = node_inputs.reshape((batch_size,
                                           node_count, feature_size))

        matrix_samples = batch_size * node_count * node_count
        matrix_inputs = np.random.randint(2, size=matrix_samples)
        matrix_inputs = matrix_inputs.reshape((batch_size,
                                               node_count, node_count))

        outputs = model.predict([node_inputs, matrix_inputs])
        self.assertEqual(outputs.shape, (batch_size, node_count, feature_units))
