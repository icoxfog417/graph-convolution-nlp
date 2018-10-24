import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gcn.graph.base_graph import BaseGraph


class SimilarityGraph(BaseGraph):

    def __init__(self, root="", graph_name="similarity_graph"):
        super().__init__(root, graph_name)

    def build(self, vocabulary,
              representation="GloVe.6B.200d", nearest_neighbor=4,
              mode="connectivity", save=True):
        # download representation
        self.storage.chakin(name=representation)

        # Make embedding matrix
        embedding = vocabulary.make_embedding(
                        self.storage.data_path("external/glove.6B.200d.txt"))

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
