import os
import zipfile
import requests
import pandas as pd


class MultiNLIData():

    def __init__(self):
        self.url = "https://s3-ap-northeast-1.amazonaws.com/dev.tech-sketch.jp/chakki/chazutsu/multinli_lm.zip"

    def download(self, path):
        resp = requests.get(self.url, stream=True)
        save_file_path = os.path.abspath(os.path.join(path, "multinli_lm.zip"))
        _dir = os.path.abspath(os.path.join(path, "multinli_lm"))

        if os.path.exists(_dir):
            return self.make_resource(_dir)

        with open(save_file_path, "wb") as f:
            chunk_size = 1024
            for data in resp.iter_content(chunk_size=chunk_size):
                f.write(data)

        with zipfile.ZipFile(save_file_path) as z:
            z.extractall(path=_dir)

        os.remove(save_file_path)
        return self.make_resource(_dir)

    def make_resource(self, multinli_dir):
        return MultiNLIResource(multinli_dir)


class MultiNLIResource():

    def __init__(self, data_dir):
        self.data_dir = data_dir

    @property
    def train_data(self):
        path = os.path.join(self.data_dir, "train_multinli.csv")
        return pd.read_csv(path, sep="\t")

    @property
    def dev_data(self):
        path = os.path.join(self.data_dir, "dev_multinli.csv")
        return pd.read_csv(path, sep="\t")
