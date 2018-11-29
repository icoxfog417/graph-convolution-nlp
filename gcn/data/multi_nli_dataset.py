import pandas as pd
import spacy
import chazutsu
from chariot.storage import Storage


class MultiNLIDataset():

    def __init__(self, root, min_word_count=3, max_word_count=25,
                 prefix=""):
        self.storage = Storage(root)
        self.nlp = spacy.load("en_core_web_sm", parser=False, entity=False)
        self.min_word_count = min_word_count
        self.max_word_count = max_word_count
        self.prefix = prefix

    def train_data(self):
        return pd.read_csv(self.processed_file("train"))

    def test_data(self):
        return pd.read_csv(self.processed_file("test"))

    @classmethod
    def labels(self):
        return ["fiction", "government", "slate", "telephone", "travel",
                "nineeleven", "facetoface", "letters", "oup", "verbatim"]

    def download(self):
        download_dir = self.storage.data_path("raw")
        matched = chazutsu.datasets.MultiNLI.matched().download(download_dir)
        mismatched = chazutsu.datasets.MultiNLI.mismatched().download(download_dir)

        for kind in ["train", "test"]:
            data = self._merge_data(matched, mismatched, kind)
            data.to_csv(self.interim_file(kind))
            preprocessed = self.preprocess(data)
            preprocessed = pd.concat([preprocessed["text"],
                                      preprocessed["label"]], axis=1)
            preprocessed.to_csv(self.processed_file(kind), index=False)
        return self

    def interim_file(self, kind):
        if self.prefix:
            p = "interim/{}_multi_nli_{}.csv".format(self.prefix, kind)
        else:
            p = "interim/multi_nli_{}.csv".format(kind)

        return self.storage.data_path(p)

    def processed_file(self, kind):
        if self.prefix:
            p = "processed/{}_multi_nli_{}.csv".format(self.prefix, kind)
        else:
            p = "processed/multi_nli_{}.csv".format(kind)

        return self.storage.data_path(p)

    def preprocess(self, df):
        # Drop duplicates
        except_d = df.drop_duplicates(["text"])

        # Count words
        word_count = except_d["text"].apply(lambda x: len(self.nlp(x)))
        except_d["word_count"] = word_count

        limited = except_d[(self.min_word_count <= except_d["word_count"]) &
                           (except_d["word_count"] <= self.max_word_count)]

        # Equalize data count
        min_count = limited["label"].value_counts().min()
        selected = limited.groupby("label").apply(lambda x: x.sample(n=min_count))
        selected = selected.drop(columns=["label", "index"]).reset_index()

        # Convert label to index
        selected["label"] = selected["label"].apply(
                                lambda x: self.labels().index(x))

        return selected

    def _merge_data(self, matched, mismatched, kind="train"):
        dataset = []
        for d in [matched, mismatched]:
            if kind == "train":
                _d = d.dev_data()
            else:
                _d = d.test_data()

            _d = pd.concat([_d["genre"], _d["sentence1"]], axis=1)
            dataset.append(_d)
        merged = pd.concat(dataset).reset_index()
        merged.rename(columns={"sentence1": "text", "genre": "label"},
                      inplace=True)
        return merged
