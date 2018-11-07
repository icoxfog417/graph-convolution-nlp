import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import numpy as np
import scipy.sparse as sp

# Disable TensorFlow GPU for parallel execution
if os.name == "nt":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.python import keras as K
from chariot.storage import Storage
from gcn.layers.graph_attention_layer import GraphAttentionLayer
from gcn.data.graph_dataset import GraphDataset


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


# Read data
root = os.path.join(os.path.dirname(__file__), "../")
storage = Storage(root)
gd = GraphDataset(root, kind="cora")
A, X, Y_train, Y_val, Y_test, idx_train, idx_val, idx_test = gd.download()

# Parameters
N = X.shape[0]                # Number of nodes in the graph
F = X.shape[1]                # Original feature dimension
n_classes = Y_train.shape[1]  # Number of classes
F_ = 8                        # Output size of first GraphAttention layer
n_attn_heads = 8              # Number of attention heads in first GAT layer
dropout_rate = 0.6            # Dropout rate (between and inside GAT layers)
l2_reg = 5e-4/2               # Factor for l2 regularization
learning_rate = 5e-3          # Learning rate for Adam
epochs = 10000                # Number of training epochs
es_patience = 100             # Patience fot early stopping
l2 = K.regularizers.l2

# Preprocessing operations
X = preprocess_features(X)
A = A + np.eye(A.shape[0])  # Add self-loops

# Model definition (as per Section 3.3 of the paper)
X_in = K.layers.Input(shape=(N, F))
A_in = K.layers.Input(shape=(N, N))

dropout1 = K.layers.Dropout(dropout_rate)(X_in)

graph_attention_1 = GraphAttentionLayer(
                        feature_units=F_,
                        attn_heads=n_attn_heads,
                        attn_heads_reduction="concat",
                        dropout_rate=dropout_rate,
                        activation="elu",
                        kernel_regularizer=l2(l2_reg),
                        attn_kernel_regularizer=l2(l2_reg))([dropout1, A_in])

dropout2 = K.layers.Dropout(dropout_rate)(graph_attention_1)
graph_attention_2 = GraphAttentionLayer(
                        n_classes,
                        attn_heads=1,
                        attn_heads_reduction="average",
                        dropout_rate=dropout_rate,
                        activation="softmax",
                        kernel_regularizer=l2(l2_reg),
                        attn_kernel_regularizer=l2(l2_reg))([dropout2, A_in])

# Build model
model = K.models.Model(inputs=[X_in, A_in], outputs=graph_attention_2)
optimizer = K.optimizers.Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss="categorical_crossentropy",
              weighted_metrics=["acc"])
model.summary()

# Callbacks
experiment_dir = storage.data_path("log/gan_experiment")
model_path = os.path.join(experiment_dir, "best_model.h5")
es_callback = K.callbacks.EarlyStopping(
                  monitor="val_weighted_acc", patience=es_patience)
tb_callback = K.callbacks.TensorBoard(log_dir=experiment_dir, batch_size=N)
mc_callback = K.callbacks.ModelCheckpoint(
                              model_path,
                              monitor="val_weighted_acc",
                              save_best_only=True,
                              save_weights_only=True)


validation_data = ((np.array([X]), np.array([A])),
                   np.array([Y_val]))
model.fit((np.array([X]), np.array([A])),
          np.array([Y_train]),
          epochs=epochs,
          batch_size=1,
          validation_data=validation_data,
          callbacks=[es_callback, tb_callback, mc_callback])

# Load best model
model.load_weights(model_path)

# Evaluate model
eval_results = model.evaluate((np.array([X]), np.array([A])),
                              np.array([Y_test]),
                              batch_size=1,
                              verbose=0)
print("Done.\n"
      "Test loss: {}\n"
      "Test accuracy: {}".format(*eval_results))
