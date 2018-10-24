import os
import numpy as np
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from chariot.storage import Storage


class DependencyGraph():

    def __init__(self, lang="en"):
        self.lang = lang
        self._parser = spacy.load(self.lang)

    def build(self, sequence):
        pass
