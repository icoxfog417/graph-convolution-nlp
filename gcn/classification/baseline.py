from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


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
