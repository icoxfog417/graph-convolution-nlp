import numpy as np
import spacy


class DependencyGraph():

    def __init__(self, lang, vocabulary):
        self.lang = lang
        self._parser = spacy.load(lang, disable=["ner", "textcat"])
        self._vocabulary = vocabulary

    def build(self, sequence, size=-1):
        words = self._vocabulary.inverse(sequence)
        sentence = " ".join(words)  # have to consider non-space-separated lang

        _size = size if size > 0 else len(sequence)
        matrix = np.zeros((_size, _size))
        tokens = self._parser(sentence)
        for token in tokens:
            # print("{} =({})=> {}".format(token.text, token.dep_, token.head.text))
            if token.i < _size and token.head.i < _size:
                matrix[token.i, token.head.i] = 1

        return matrix

    def batch_build(self, sequences, size=-1):
        matrices = [self.build(s, size) for s in sequences]
        return np.array(matrices)
