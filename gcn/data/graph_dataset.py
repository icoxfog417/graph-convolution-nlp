import os
import pickle as pkl
import zipfile
import requests
import numpy as np
import scipy.sparse as sp
import networkx as nx
from chariot.storage import Storage


class GraphDataset():

    def __init__(self, root, kind="cora"):
        self.storage = Storage(root)
        self.kind = kind
        self.download_url = "https://s3-ap-northeast-1.amazonaws.com/dev.tech-sketch.jp/chakki/public/graph/"  # noqa
        if kind == "cora":
            self.download_url += "cora.zip"
        elif kind == "citeseer":
            self.download_url += "citeseer.zip"
        elif kind == "pubmed":
            self.download_url += "pubmed.zip"
        else:
            raise Exception("Graph dataset {} is not supported.".format(kind))

    @property
    def data_root(self):
        return self.storage.data_path("raw/{}".format(self.kind))

    @property
    def download_file_path(self):
        return self.storage.data_path("raw/{}.zip".format(self.kind))

    def download(self, return_mask=True):
        # Check downloaded file
        if os.path.isdir(self.data_root):
            print("{} dataset is already downloaded.".format(self.kind))
            return self.load(return_mask)

        # Download dataset
        resp = requests.get(self.download_url, stream=True)
        with open(self.download_file_path, "wb") as f:
            chunk_size = 1024
            for data in resp.iter_content(chunk_size=chunk_size):
                f.write(data)

        # Expand file
        with zipfile.ZipFile(self.download_file_path) as z:
            z.extractall(path=self.data_root)
        os.remove(self.download_file_path)

        return self.load(return_mask)

    def load(self, return_mask):
        """
        Loads input data (reference from: https://github.com/tkipf/gcn/blob/master/gcn/utils.py)
        ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
            (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
        ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
        ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
        ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
            object;
        ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
        All objects above must be saved using python pickle module.
        :param dataset_str: Dataset name
        :return: All data input files loaded (as well the training/test data).
        """

        names = ["x", "y", "tx", "ty", "allx", "ally", "graph", "test.index"]
        objects = []
        for n in names:
            file_path = os.path.join(self.data_root,
                                     "ind.{}.{}".format(self.kind, n))

            if n != "test.index":
                with open(file_path, "rb") as f:
                    objects.append(pkl.load(f, encoding="latin1"))
            else:
                with open(file_path, encoding="latin1") as f:
                    lines = f.readlines()
                    indices = [int(ln.strip()) for ln in lines]
                objects.append(indices)

        x, y, tx, ty, allx, ally, graph, test_idx = tuple(objects)
        test_idx_range = np.sort(test_idx)

        if self.kind == "citeseer":
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx), max(test_idx)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx, :] = labels[test_idx_range, :]

        idx_test = test_idx_range
        idx_train = np.array(range(len(y)))
        idx_val = np.array(range(len(y), len(y)+500))

        if return_mask:
            train_mask = self.sample_mask(idx_train, labels.shape[0])
            val_mask = self.sample_mask(idx_val, labels.shape[0])
            test_mask = self.sample_mask(idx_test, labels.shape[0])

            y_train = np.zeros(labels.shape)
            y_val = np.zeros(labels.shape)
            y_test = np.zeros(labels.shape)
            y_train[train_mask, :] = labels[train_mask, :]
            y_val[val_mask, :] = labels[val_mask, :]
            y_test[test_mask, :] = labels[test_mask, :]

            return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
        else:

            y_train = labels[idx_train, :]
            y_val = labels[idx_val, :]
            y_test = labels[idx_test, :]
            return adj, features, y_train, y_val, y_test, idx_train, idx_val, idx_test

    def sample_mask(self, idx, length):
        """Create mask."""
        mask = np.zeros(length)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)
