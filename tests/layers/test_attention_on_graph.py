import unittest
import numpy as np
from scipy.spatial import distance_matrix
from tensorflow.python import keras as K
from tests.layers.simple_attention_layer import SimpleAttentionLayer


class TestAttentionOnGraph(unittest.TestCase):

    def test_attention_learning(self):
        exp1 = self.run_attention_learning("column", "add", False)  # original
        exp2 = self.run_attention_learning("column", "add", True)
        exp3 = self.run_attention_learning("column", "concat", True)
        exp4 = self.run_attention_learning("row", "add", False)
        exp5 = self.run_attention_learning("row", "add", True)
        exp6 = self.run_attention_learning("row", "concat", True)

        for acc in [exp2, exp3, exp4, exp5, exp6]:
            # original method should be most accurate
            self.assertGreater(exp1, acc)

    def run_attention_learning(self, node_axis, merge_method,
                               use_attention_kernel):
        node_count = 10
        feature_size = 2
        feature_units = 2
        problem_count = 10000
        validation_count = 5

        model = self.make_simple_attention_network(
                    node_count, feature_size, feature_units,
                    node_axis, merge_method,
                    use_attention_kernel)

        model.compile(loss="categorical_crossentropy", optimizer="adam",
                      metrics=["accuracy"])

        last_accs = []
        for i in range(validation_count):
            params = self.make_problems(node_count, feature_size,
                                        feature_units, problem_count)
            node_inputs, matrix_inputs, answers, attn_answers = params

            metrics = model.fit([node_inputs, matrix_inputs], attn_answers,
                                validation_split=0.2, epochs=5)
            acc = metrics.history["val_acc"][-1]
            last_accs.append(acc)

        def calc_baseline_acc(A, label):
            x = np.random.normal(size=A.shape) * A
            x_exp = np.exp(x)
            x = x_exp / np.sum(x_exp, axis=-1, keepdims=True)
            match = np.equal(np.argmax(label, axis=-1),
                             np.argmax(x_exp, axis=-1),)
            count = A.shape[0] * A.shape[1]
            acc = np.sum(match) / count
            return acc

        baseline_acc = calc_baseline_acc(matrix_inputs, attn_answers)
        method = "Merge: {} Node: {} Attention: {}".format(
            merge_method, node_axis, use_attention_kernel)
        if merge_method == "add" and node_axis == "column" and \
           not use_attention_kernel:
            method += " (original)"

        print(method)
        acc = np.mean(last_accs)
        print("\t acc: {}(+/-{}) (baseline {})".format(
            acc, np.std(last_accs), baseline_acc))
        return acc

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

            answers.append(n[target_index])
            attn = np.zeros(m.shape)
            attn[np.arange(len(attn)), target_index] = 1
            attention_answers.append(attn)

        answers = np.array(answers)
        attention_answers = np.array(attention_answers)

        return node_inputs, matrix_inputs, answers, attention_answers

    def make_simple_attention_network(self, node_count,
                                      feature_size, feature_units,
                                      node_axis, merge_method,
                                      use_attention_kernel):

        nodes = K.layers.Input(shape=(node_count, feature_size))
        matrix = K.layers.Input(shape=(node_count, node_count))
        layer = SimpleAttentionLayer(feature_units=feature_units,
                                     node_axis=node_axis,
                                     merge_method=merge_method,
                                     use_attention_kernel=use_attention_kernel,
                                     return_attention=True)

        _, attn = layer([nodes, matrix])
        model = K.models.Model(inputs=[nodes, matrix], outputs=attn)
        return model
