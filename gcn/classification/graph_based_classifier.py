import numpy as np
from tensorflow.python import keras as K
import tensorflow as tf
from gcn.layers.graph_attention_layer import GraphAttentionLayer
from gcn.util import gpu_enable


class GraphBasedClassifier():

    def __init__(self, vocab_size, graph_size,
                 embedding_size=100, hidden_size=100,
                 head_types=("concat",), heads=1, dropout=0.6,
                 node_level_bias=False, with_attention=True,
                 lstm=None, bidirectional=False):

        self.vocab_size = vocab_size
        self.graph_size = graph_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.head_types = head_types
        self.heads = heads
        self.dropout = dropout
        self.node_level_bias = node_level_bias
        self.with_attention = with_attention
        self.lstm = lstm
        self.bidirectional = bidirectional
        self.model = None
        self._attention = None
        self.preprocessor = None

    def build(self, num_classes, preprocessor=None):
        X_in = K.layers.Input(shape=(self.graph_size,))
        A_in = K.layers.Input(shape=(self.graph_size, self.graph_size))
        self.preprocessor = preprocessor

        embedding = K.layers.Embedding(input_dim=self.vocab_size,
                                       output_dim=self.embedding_size,
                                       input_length=self.graph_size,
                                       embeddings_regularizer=K.regularizers.l2(),
                                       name="embedding",
                                       mask_zero=True)
        vectors = embedding(X_in)
        _vectors = K.layers.Dropout(self.dropout)(vectors)

        def lstm(return_sequences):
            # CuDNNLSTM does not support mask.
            # layer = K.layers.CuDNNLSTM if gpu_enable() else K.layers.LSTM
            layer = K.layers.LSTM
            _lstm = layer(self.hidden_size, return_sequences=return_sequences,
                          dropout=self.dropout, recurrent_dropout=self.dropout)
            if self.bidirectional:
                _lstm = K.layers.Bidirectional(_lstm, merge_mode="concat")
            return _lstm

        if self.lstm is not None and self.lstm == "before":
            _vectors = lstm(return_sequences=True)(_vectors)

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
                        return_attention=True,
                        node_level_bias=self.node_level_bias)
            _vectors, attention = gh([_vectors, A_in])
            attentions.append(attention)

        if self.lstm is not None and self.lstm == "after":
            merged = lstm(return_sequences=False)(_vectors)
        else:
            merged = K.layers.Lambda(lambda x: K.backend.sum(x, axis=1))(_vectors)

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
        attentions = self._attention.predict(_x)
        if len(self.head_types) == 1:
            attentions = [attentions]

        # batch, layer, head, node_size, node_size
        attentions = np.array(attentions)
        attentions = np.transpose(attentions, (1, 0, 2, 3, 4))
        return attentions
