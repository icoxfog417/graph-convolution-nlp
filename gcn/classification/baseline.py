from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from tensorflow.python import keras as K
from gcn.util import gpu_enable


class TfidfClassifier():

    def __init__(self, max_df=1.0, min_df=1, vocabulary=None):
        self.vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df,
                                          vocabulary=vocabulary)
        self.classifier = LogisticRegression(penalty="l1")
        self.model = Pipeline([("vectorizer", self.vectorizer),
                               ("classifier", self.classifier)])

    def fit(self, x, y, cv=5):
        scores = cross_val_score(self.model, x, y, cv=cv, scoring="f1_micro")
        self.model.fit(x, y)
        return scores


class LSTMClassifier():

    def __init__(self, vocab_size, embedding_size=100, hidden_size=100,
                 layers=1, dropout=0.5):

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.dropout = dropout
        self.model = None

    def build(self, num_classes):
        # Build the model
        model = K.Sequential()
        embedding = K.layers.Embedding(input_dim=self.vocab_size,
                                       output_dim=self.embedding_size)
        model.add(embedding)
        model.add(K.layers.Dropout(self.dropout))
        rnn_layer = K.layers.CuDNNLSTM if gpu_enable() else K.layers.LSTM
        for layer in range(self.layers):
            model.add(rnn_layer(self.hidden_size))

        model.add(K.layers.Dropout(self.dropout))
        model.add(K.layers.Dense(num_classes, activation="softmax"))

        self.model = model
