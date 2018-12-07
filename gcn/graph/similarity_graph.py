import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from chariot.storage import Storage


class SimilarityGraph():

    def __init__(self, vocabulary, nearest_neighbor=4, mode="connectivity",
                 representation="GloVe.6B.200d", root=""):
        self.vocabulary = vocabulary
        self.nearest_neighbor = nearest_neighbor
        self.mode = mode
        self.representation = representation
        default_root = os.path.join(os.path.dirname(__file__), "../../")
        _root = root if root else default_root

        self.storage = Storage(_root)
        self.embedding = None

    def build(self, sequence, size=-1):
        if 0 < size < self.nearest_neighbor:
            raise Exception("Matrix size is not enough for neighbors.")

        if self.embedding is None:
            # download representation
            self.storage.chakin(name=self.representation)

            # Make embedding matrix
            file_path = "external/{}.txt".format(self.representation.lower())
            self.embedding = self.vocabulary.make_embedding(
                                self.storage.data_path(file_path))

        vectors = np.vstack([self.embedding[s] for s in sequence])
        matrix = self._build(vectors, size)
        return matrix

    def _build(self, vectors, size=-1):
        _size = size if size > 0 else len(vectors)
        similarity = cosine_similarity(vectors[:_size])
        similarity -= np.eye(_size)  # exclude similarity to self
        top_k = np.argsort(-similarity, axis=1)[:, :self.nearest_neighbor]

        matrix = np.zeros((_size, _size))
        for i, top in enumerate(top_k):
            if self.mode == "connectivity":
                matrix[i, top] = 1
            else:
                matrix[i, top] = similarity[i, top]

        return matrix

    def batch_build(self, sequences, size=-1):
        matrices = [self.build(s, size) for s in sequences]
        return np.array(matrices)
