import numpy as np
import spacy


class StaticGraph():

    def __init__(self, lang, kind="previous", fill=True):
        self.lang = lang
        self._parser = spacy.load(self.lang, disable=["ner", "textcat"])
        self.kind = kind
        self.fill = fill

    def get_nodes(self, sentence):
        return [t.text for t in self._parser(sentence)]

    def build(self, sentence, size=-1):
        nodes = self.get_nodes(sentence)
        _size = size if size > 0 else len(nodes)
        if self.fill:
            func = lambda s, k=0: np.tril(np.ones((s, s)), k)
        else:
            func = np.eye

        if self.kind == "self":
            return func(_size)
        elif self.kind == "previous":
            return func(_size, k=-1)

    def batch_build(self, sentences, size=-1):
        matrices = [self.build(s, size) for s in sentences]
        return np.array(matrices)
