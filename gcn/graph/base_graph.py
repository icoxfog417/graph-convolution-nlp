import os
import numpy as np
from chariot.storage import Storage


class BaseGraph():

    def __init__(self, root="", graph_name="graph"):
        default_root = os.path.join(os.path.dirname(__file__), "../../")
        _root = root if root else default_root

        self.storage = Storage(_root)
        self.graph_name = graph_name

    def save(self, graph):
        np.save(self.path, graph)

    def load(self):
        if os.path.exists(self.path):
            return np.load(self.path)
        else:
            return None

    @property
    def path(self):
        path = "interim/{}.npy".format(self.graph_name)
        return self.storage.data_path(path)
