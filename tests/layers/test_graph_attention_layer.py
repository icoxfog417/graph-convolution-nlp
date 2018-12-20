import unittest
import numpy as np
from scipy.spatial import distance_matrix
from tensorflow.python import keras as K
from gcn.layers.graph_attention_layer_before import GraphAttentionLayer as GraphAttentionLayerB
from gcn.layers.graph_attention_layer import GraphAttentionLayer


class TestGraphAttentionLayer(unittest.TestCase):
    TEST_PATTERN = (10, 2, 2)

    def test_forward(self):
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

    def test_training(self):
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
                    merge="average")
        model.compile(loss="mse", optimizer="adam")
        metrics = model.fit([node_inputs, matrix_inputs], answers,
                            validation_split=0.3,
                            epochs=50)
        last_loss = metrics.history["val_loss"][-1]
        min_loss = np.min(metrics.history["val_loss"])
        self.assertEqual(last_loss, min_loss)

    def test_attention(self):
        node_count, feature_size, feature_units = self.TEST_PATTERN

        def make_model():
            model, model_attn = self.make_graph_attention_network(
                                    node_count, feature_size, feature_units,
                                    return_attention=True)
            return model, model_attn

        loss, hit_prob = self._test_attention(make_model,
                                              node_count,
                                              feature_size, feature_units,
                                              header="GAL After")

        self.assertGreater(hit_prob, 0.7)

    def test_attention_before(self):
        node_count, feature_size, feature_units = self.TEST_PATTERN

        def make_model():
            model, model_attn = self.make_graph_attention_network(
                                    node_count, feature_size, feature_units,
                                    return_attention=True, before=True)
            return model, model_attn

        loss, hit_prob = self._test_attention(make_model,
                                              node_count,
                                              feature_size, feature_units,
                                              header="GAL Before")
        self.assertGreater(hit_prob, 0.7)

    def test_attention_theoretical(self):
        node_count, feature_size, feature_units = self.TEST_PATTERN

        def make_model():
            model, model_attn = self.make_simple_attention_network(
                                    node_count, feature_size, feature_units,
                                    return_attention=True)
            return model, model_attn

        loss, hit_prob = self._test_attention(make_model,
                                              node_count,
                                              feature_size, feature_units,
                                              header="Theoretical Attention")
        self.assertGreater(hit_prob, 0.7)

    def _test_attention(self, make_model,
                        node_count, feature_size, feature_units,
                        problem_count=10000, varidation_count=5,
                        header=""):

        losses = []
        hit_probs = []
        for i in range(varidation_count):
            model, model_attn = make_model()
            params = self.make_problems(node_count, feature_size,
                                        feature_units, problem_count)
            node_inputs, matrix_inputs, answers, attn_answers = params

            model.compile(loss="mse", optimizer="adam")
            model.fit([node_inputs, matrix_inputs], answers,
                      validation_split=0.3, epochs=20)

            attentions = model_attn.predict([node_inputs, matrix_inputs])

            if len(attentions.shape) == 4:
                attentions = attentions[:, 0, :, :]  # attention of head 0

            loss, hit_prob = self.calculate_attention_loss(
                                attentions, attn_answers)
            losses.append(loss)
            hit_probs.append(hit_prob)

        loss = np.mean(losses)
        hit_prob = np.mean(hit_probs)
        if header:
            print(header)
        print("\t loss: {}(+/-{}), hit_prob:{} (+/-{}).".format(
            loss, np.std(losses), hit_prob, np.std(hit_prob)))
        return loss, hit_prob

    def make_problems(self, node_count, feature_size, feature_units,
                      problem_count):
        """
        Make task to extract the nearest node from neighbors.
        """

        node_samples = problem_count * node_count * feature_size
        node_inputs = np.random.uniform(high=10, size=node_samples).reshape(
                        (problem_count, node_count, feature_size))

        matrix_samples = problem_count * node_count * node_count
        matrix_inputs = np.random.randint(2, size=matrix_samples).reshape(
                            (problem_count, node_count, node_count))

        answers = []
        attention_answers = []
        for n, m in zip(node_inputs, matrix_inputs):
            distance = distance_matrix(n, n)
            mask = 10e9 * (1.0 - m)
            target_index = np.argmin(distance * m + mask, axis=1)

            if feature_size == feature_units:
                answers.append(n[target_index])
            else:
                answers.append(n[target_index][:, :feature_units])

            attn = np.zeros(m.shape)
            attn[np.arange(len(attn)), target_index] = 1
            attention_answers.append(attn)

        answers = np.array(answers)
        attention_answers = np.array(attention_answers)

        return node_inputs, matrix_inputs, answers, attention_answers

    def make_graph_attention_network(self, node_count,
                                     feature_size, feature_units,
                                     head=1, merge="average",
                                     return_attention=False,
                                     before=False):

        nodes = K.layers.Input(shape=(node_count, feature_size))
        matrix = K.layers.Input(shape=(node_count, node_count))

        if before:
            GAL = GraphAttentionLayerB
        else:
            GAL = GraphAttentionLayer

        layer = GAL(feature_units=feature_units,
                    attn_heads=head,
                    attn_heads_reduction=merge,
                    dropout_rate=0.0,
                    return_attention=return_attention,
                    node_level_bias=True)

        if return_attention:
            output, attn = layer([nodes, matrix])
        else:
            output = layer([nodes, matrix])

        model = K.models.Model(inputs=[nodes, matrix], outputs=output)
        if return_attention:
            model_attn = K.models.Model(inputs=[nodes, matrix], outputs=attn)
            return model, model_attn
        else:
            return model

    def make_simple_attention_network(self, node_count,
                                      feature_size, feature_units,
                                      return_attention=False):

        from tests.layers.simple_attention_layer import SimpleAttentionLayer

        nodes = K.layers.Input(shape=(node_count, feature_size))
        matrix = K.layers.Input(shape=(node_count, node_count))
        layer = SimpleAttentionLayer(feature_units=feature_units,
                                     return_attention=return_attention)

        if return_attention:
            output, attn = layer([nodes, matrix])
            attn = attn
        else:
            output = layer([nodes, matrix])

        model = K.models.Model(inputs=[nodes, matrix], outputs=output)
        if return_attention:
            model_attn = K.models.Model(inputs=[nodes, matrix], outputs=attn)
            return model, model_attn
        else:
            return model

    def calculate_attention_loss(self, predicted, answers):
        loss = 0
        hit_prob = 0

        for p, a in zip(predicted, answers):
            norm = np.linalg.norm(p * a - a)
            hits = np.sum(np.equal(np.argmax(p, axis=1),
                                   np.argmax(a, axis=1)))
            hit_prob += hits / len(p)
            loss += norm
        loss = loss / len(predicted)
        hit_prob = hit_prob / len(predicted)
        return loss, hit_prob
