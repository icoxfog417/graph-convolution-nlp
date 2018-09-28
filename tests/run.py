import os
import chazutsu
from chariot.storage import Storage
import chariot.transformer as ct
from chariot.preprocessor import Preprocessor


# Download Text8
data_dir = os.path.join(os.path.dirname(__file__), "../../")
storage = Storage(data_dir)
r = chazutsu.datasets.Text8().download(storage.data_path("raw"))


preprocessor = Preprocessor(
                  text_transformers=[ct.text.UnicodeNormalizer()],
                  tokenizer=ct.Tokenizer(lang=None),
                  vocabulary=ct.Vocabulary())

print(r.train_data()["sentence"])
#preprocessor.fit(r.train_data()[:100])
