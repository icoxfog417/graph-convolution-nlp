import unittest
import numpy as np
from tensorflow.python import keras as K
from gcn.layers.graph_attention_layer import GraphAttentionLayer


class TestGraphAttentionLayer(unittest.TestCase):

    def xtest_forward(self):
        node_count = 12
        feature_size = 10
        feature_units = 8
        head = 3

        batch_size = 32
        node_samples = batch_size * node_count * feature_size
        node_inputs = np.random.uniform(size=node_samples)
        node_inputs = node_inputs.reshape((batch_size,
                                           node_count, feature_size))

        matrix_samples = batch_size * node_count * node_count
        matrix_inputs = np.random.randint(2, size=matrix_samples)
        matrix_inputs = matrix_inputs.reshape((batch_size,
                                               node_count, node_count))

        concat_model = self.make_graph_attention_network(
                            node_count, feature_size, feature_units,
                            head=head, merge="concat")
        outputs = concat_model.predict([node_inputs, matrix_inputs])
        self.assertEqual(outputs.shape, (batch_size, node_count,
                                         feature_units * head))

        mean_model = self.make_graph_attention_network(
                            node_count, feature_size, feature_units,
                            head=head, merge="average")
        outputs = mean_model.predict([node_inputs, matrix_inputs])
        self.assertEqual(outputs.shape, (batch_size, node_count,
                                         feature_units))

    def xtest_training(self):
        node_count = 4
        feature_size = 3
        feature_units = 1
        problem_count = 1000

        node_inputs, matrix_inputs, answers, _ = self.make_problems(
                                                    node_count, feature_size,
                                                    feature_units,
                                                    problem_count)

        model = self.make_graph_attention_network(
                    node_count, feature_size, feature_units,
                    merge="average", activation=True)
        model.compile(loss="mse", optimizer="adam")
        metrics = model.fit([node_inputs, matrix_inputs], answers,
                            validation_split=0.3,
                            epochs=50)
        last_loss = metrics.history["val_loss"][-1]
        self.assertLess(last_loss, 1e-1)

    def test_attention(self):
        node_count = 5
        feature_size = 8
        feature_units = 3
        problem_count = 1000

        params = self.make_problems(node_count, feature_size,
                                    feature_units, problem_count)
        node_inputs, matrix_inputs, answers, attn_answers = params

        model, model_attn = self.make_graph_attention_network(
                                node_count, feature_size, feature_units,
                                merge="average", activation=True,
                                return_attention=True)
        model.compile(loss="mse", optimizer="adam")
        model.fit([node_inputs, matrix_inputs], answers,
                  validation_split=0.3, epochs=50)

        test_samples = 10
        sample_index = np.random.randint(problem_count, size=test_samples)
        attentions = model_attn.predict([node_inputs[sample_index],
                                        matrix_inputs[sample_index]])

        print(attentions[1])
        print(attn_answers[1])
        loss = 0
        for i in range(test_samples):
            norm = np.linalg.norm(attn_answers[i] - attentions[i])
            loss += norm
        loss = loss / test_samples
        self.assertLess(loss, 1e-1)

    def make_problems(self, node_count, feature_size, feature_units,
                      problem_count):
        """
        Problem: attention to maximum neighbor.
        Node         Matrix      Answer
        [1, 9, 3]     [0, 1, 1    [9, 3, 1]
                       1, 0, 1,
                       1, 0, 0]
        """

        node_samples = problem_count * node_count * feature_size
        node_inputs = np.random.uniform(size=node_samples).reshape(
                        (problem_count, node_count, feature_size))

        matrix_samples = problem_count * node_count * node_count
        matrix_inputs = np.random.randint(2, size=matrix_samples).reshape(
                            (problem_count, node_count, node_count))

        answers = []
        attention_answers = []
        for n, m in zip(node_inputs, matrix_inputs):
            values = np.linalg.norm(n, axis=1)
            max_index = np.argmax(values * m, axis=1)
            answer = values[max_index]
            answers.append(answer.reshape(len(n), 1))
            attn = np.zeros(m.shape)
            attn[np.arange(len(attn)), max_index] = 1
            attention_answers.append(attn)

        answers = np.array(answers)
        attention_answers = np.array(attention_answers)

        return node_inputs, matrix_inputs, answers, attention_answers

    def make_graph_attention_network(self, node_count,
                                     feature_size, feature_units,
                                     head=1, merge="concat",
                                     activation=False,
                                     return_attention=False):
        nodes = K.layers.Input(shape=(node_count, feature_size))
        matrix = K.layers.Input(shape=(node_count, node_count))
        layer = GraphAttentionLayer(feature_units=feature_units,
                                    attn_heads=head,
                                    attn_heads_reduction=merge,
                                    return_attention=return_attention)

        if return_attention:
            output, attn = layer([nodes, matrix])
        else:
            output = layer([nodes, matrix])

        if activation:
            output = K.layers.Dense(1)(output)

        model = K.models.Model(inputs=[nodes, matrix], outputs=output)
        if return_attention:
            model_attn = K.models.Model(inputs=[nodes, matrix], outputs=attn)
            return model, model_attn
        else:
            return model
