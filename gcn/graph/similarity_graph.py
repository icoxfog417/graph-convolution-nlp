import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from chariot.storage import Storage


class SimilarityGraph():

    def __init__(self, root="", graph_name="similarity_graph"):
        default_root = os.path.join(os.path.dirname(__file__), "../../")
        _root = root if root else default_root

        self.storage = Storage(_root)
        self.graph_name = graph_name

    @property
    def path(self):
        path = "interim/{}.npy".format(self.graph_name)
        return self.storage.data_path(path)

    def save(self, graph):
        np.save(self.path, graph)

    def load(self):
        if os.path.exists(self.path):
            return np.load(self.path)
        else:
            return None

    def build(self, vocabulary,
              representation="GloVe.6B.200d", nearest_neighbor=4,
              mode="connectivity", save=True):
        # download representation
        self.storage.chakin(name=representation)

        # Make embedding matrix
        file_path = "external/{}.txt".format(representation.lower())
        embedding = vocabulary.make_embedding(
                        self.storage.data_path(file_path))

        graph = self._build(embedding, nearest_neighbor, mode)
        if save:
            self.save(graph)
        return graph

    def _build(self, vectors, nearest_neighbor, mode):
        # Normalize embedding to emulate cosine similarity
        # by euclidean distance
        # https://cmry.github.io/notes/euclidean-v-cosine

        similarity = cosine_similarity(vectors)
        similarity -= np.eye(len(vectors))
        top_k = np.argsort(-similarity, axis=1)[:, :nearest_neighbor]

        matrix = np.zeros(similarity.shape)
        for i, top in enumerate(top_k):
            if mode == "connectivity":
                matrix[i, top] = 1
            else:
                matrix[i, top] = similarity[i, top]

        return matrix
