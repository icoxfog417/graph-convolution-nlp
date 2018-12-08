import numpy as np
from tensorflow.python import keras as K
from gcn.layers.graph_attention_layer import GraphAttentionLayer
from gcn.util import gpu_enable


class GraphBasedClassifier():

    def __init__(self, vocab_size, graph_size,
                 embedding_size=100, hidden_size=100,
                 head_types=("concat",), heads=1, dropout=0.5,
                 with_attention=True, with_lstm="before"):

        self.vocab_size = vocab_size
        self.graph_size = graph_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.head_types = head_types
        self.heads = heads
        self.dropout = dropout
        self.with_attention = with_attention
        self.with_lstm = with_lstm
        self.model = None
        self._attention = None

    def build(self, num_classes, preprocessor=None):
        X_in = K.layers.Input(shape=(self.graph_size,))
        A_in = K.layers.Input(shape=(self.graph_size, self.graph_size))
        self.preprocessor = preprocessor

        embedding = K.layers.Embedding(input_dim=self.vocab_size,
                                       output_dim=self.embedding_size,
                                       input_length=self.graph_size,
                                       embeddings_regularizer=K.regularizers.l2())
        vectors = embedding(X_in)
        _vectors = K.layers.Dropout(self.dropout)(vectors)
        if self.with_lstm == "before":
            o = K.layers.LSTM(self.hidden_size,
                              return_sequences=True)(_vectors)
            _vectors = o

        attentions = []
        for ht in self.head_types:
            gh = GraphAttentionLayer(
                        feature_units=self.hidden_size,
                        attn_heads=self.heads,
                        attn_heads_reduction=ht,
                        dropout_rate=self.dropout,
                        kernel_regularizer=K.regularizers.l2(),
                        attention=self.with_attention,
                        attn_kernel_regularizer=K.regularizers.l2(),
                        return_attention=True)
            _vectors, attention = gh([_vectors, A_in])
            attentions.append(attention)

        if self.with_lstm == "after":
            o, h, c = K.layers.LSTM(self.hidden_size, return_sequences=True,
                                    return_state=True)(_vectors)
            _vectors = c

        merged = K.layers.Lambda(lambda x: K.backend.mean(x, axis=1))(_vectors)
        probs = K.layers.Dense(num_classes, activation="softmax")(merged)

        self.model = K.models.Model(inputs=[X_in, A_in], outputs=probs)
        self._attention = K.models.Model(inputs=[X_in, A_in],
                                         outputs=attentions)

    def predict(self, x):
        preds = self.predict_proba(x)
        return np.argmax(preds, axis=1)

    def predict_proba(self, x):
        _x = x if self.preprocessor is None else self.preprocessor(x)
        return self.model.predict(_x)

    def show_attention(self, x):
        _x = x if self.preprocessor is None else self.preprocessor(x)
        return self._attention.predict(_x)
