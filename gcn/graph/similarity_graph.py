import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from chariot.storage import Storage
from chariot.resource.word_vector import WordVector


class SimilarityGraph():

    def __init__(self, lang, nearest_neighbor=4, threshold=0.3,
                 mode="similarity", representation="GloVe.6B.100d", root=""):
        self.lang = lang
        self._parser = spacy.load(self.lang, disable=["ner", "textcat"])
        self.nearest_neighbor = nearest_neighbor
        self.threshold = threshold
        self.mode = mode
        self.representation = representation
        default_root = os.path.join(os.path.dirname(__file__), "../../")
        _root = root if root else default_root

        self.storage = Storage(_root)
        self.key_vector = {}
        self._unknown = None

    def get_nodes(self, sentence):
        return [t.text for t in self._parser(sentence)]

    def build(self, sentence, size=-1):
        if 0 < size < self.nearest_neighbor:
            raise Exception("Matrix size is not enough for neighbors.")

        if len(self.key_vector) == 0:
            # download representation
            self.storage.chakin(name=self.representation)

            # Make embedding matrix
            file_path = "external/{}.txt".format(self.representation.lower())
            wv = WordVector(self.storage.data_path(file_path))
            self.key_vector = wv.load()

            for k in self.key_vector:
                self._unknown = np.zeros(len(self.key_vector[k]))
                break

        tokens = self._parser(sentence)
        vectors = []
        for t in tokens:
            if t.text in self.key_vector:
                vectors.append(self.key_vector[t.text])
            else:
                vectors.append(self._unknown)

        vectors = np.vstack(vectors)
        matrix = self._build(vectors, size)
        return matrix

    def _build(self, vectors, size=-1):
        _size = size if size > 0 else len(vectors)
        similarity = cosine_similarity(vectors[:_size])
        similarity -= np.eye(similarity.shape[0])  # exclude similarity to self
        top_k = np.argsort(-similarity, axis=1)[:, :self.nearest_neighbor]

        matrix = np.zeros((_size, _size))
        for i, top in enumerate(top_k):
            _top = np.array([t for t in top
                             if np.abs(similarity[i, t]) >= self.threshold])

            if len(_top) == 0:
                continue

            if self.mode == "connectivity":
                matrix[i, _top] = 1
            else:
                matrix[i, _top] = similarity[i, _top]

        return matrix

    def batch_build(self, sentences, size=-1):
        matrices = [self.build(s, size) for s in sentences]
        return np.array(matrices)
