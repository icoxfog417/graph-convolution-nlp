import numpy as np
import spacy


class DependencyGraph():

    def __init__(self, lang):
        self.lang = lang
        self._parser = spacy.load(self.lang, disable=["ner", "textcat"])

    def get_nodes(self, sentence):
        return [t.text for t in self._parser(sentence)]

    def build(self, sentence, size=-1, return_label=False):
        tokens = self._parser(sentence)
        _size = size if size > 0 else len(tokens)
        matrix = np.zeros((_size, _size))
        if return_label:
            matrix = [[""] * matrix.shape[1] for r in range(matrix.shape[0])]
        for token in tokens:
            # print("{} =({})=> {}".format(token.text, token.dep_, token.head.text))
            if not token.dep_:
                raise Exception("Dependency Parse does not work well.")

            if token.i < _size and token.head.i < _size:
                v = token.dep_ if return_label else 1
                matrix[token.i][token.head.i] = v

        return matrix

    def batch_build(self, sentences, size=-1):
        matrices = [self.build(s, size) for s in sentences]
        return np.array(matrices)
