import numpy as np
import spacy


class DependencyGraph():

    def __init__(self, lang, vocabulary):
        self.lang = lang
        self._parser = spacy.load(self.lang, disable=["ner", "textcat"])
        self.vocabulary = vocabulary

    def build(self, sequence, size=-1, return_label=False):
        words = self.vocabulary.inverse(sequence)
        sentence = " ".join(words)  # have to consider non-space-separated lang

        _size = size if size > 0 else len(sequence)
        matrix = np.zeros((_size, _size))
        if return_label:
            matrix = [[""] * matrix.shape[1] for r in range(matrix.shape[0])]
        tokens = self._parser(sentence)
        for token in tokens:
            # print("{} =({})=> {}".format(token.text, token.dep_, token.head.text))
            if not token.dep_:
                raise Exception("Dependency Parse does not work well.")

            if token.i < _size and token.head.i < _size:
                v = token.dep_ if return_label else 1
                matrix[token.i][token.head.i] = v

        return matrix

    def batch_build(self, sequences, size=-1):
        matrices = [self.build(s, size) for s in sequences]
        return np.array(matrices)
