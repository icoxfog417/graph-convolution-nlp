import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import numpy as np
import scipy.sparse as sp

# Disable TensorFlow GPU to avoid memory error
if os.name == "nt":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.python import keras as K
from chariot.storage import Storage
from gcn.data.graph_dataset import GraphDataset


"""
Evaluation script is ported from
https://github.com/danielegrattarola/keras-gat/blob/master/examples/gat.py
"""


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def run_experiment(original=True, attention=True):
    if original:
        from gcn.layers.graph_attention_layer_original import GraphAttentionLayer
    else:
        from gcn.layers.graph_attention_layer import GraphAttentionLayer

    # Read data
    root = os.path.join(os.path.dirname(__file__), "../../")
    storage = Storage(root)
    gd = GraphDataset(root, kind="cora")
    data = gd.download(return_mask=original)
    A, X, Y_train, Y_val, Y_test, idx_train, idx_val, idx_test = data

    # Parameters
    N = X.shape[0]                # Number of nodes in the graph
    F = X.shape[1]                # Original feature dimension
    n_classes = Y_train.shape[1]  # Number of classes
    F_ = 8                        # Output size of first GraphAttention layer
    n_attn_heads = 8              # Number of attention heads in first GAT layer
    dropout_rate = 0.6            # Dropout rate (between and inside GAT layers)
    l2_reg = 5e-4/2               # Factor for l2 regularization
    learning_rate = 5e-3          # Learning rate for Adam
    epochs = 120                  # Number of training epochs
    es_patience = 100             # Patience fot early stopping
    l2 = K.regularizers.l2
    node_size = 32

    # Preprocessing operations
    X = preprocess_features(X)
    A = A + np.eye(A.shape[0])  # Add self-loops

    # Model definition (as per Section 3.3 of the paper)
    if original:
        X_in = K.layers.Input(shape=(F,))
        A_in = K.layers.Input(shape=(N,))    
    else:
        X_in = K.layers.Input(shape=(N, F))
        A_in = K.layers.Input(shape=(N, N))

    I_in = K.layers.Input(shape=(node_size,), dtype="int32")

    dropout1 = K.layers.Dropout(dropout_rate)(X_in)

    graph_attention_1 = GraphAttentionLayer(
                            feature_units=F_,
                            attn_heads=n_attn_heads,
                            attn_heads_reduction="concat",
                            dropout_rate=dropout_rate,
                            activation="elu",
                            kernel_regularizer=l2(l2_reg),
                            attention=attention,
                            attn_kernel_regularizer=l2(l2_reg))([dropout1, A_in])

    dropout2 = K.layers.Dropout(dropout_rate)(graph_attention_1)
    graph_attention_2 = GraphAttentionLayer(
                            n_classes,
                            attn_heads=1,
                            attn_heads_reduction="average",
                            dropout_rate=dropout_rate,
                            activation="softmax",
                            kernel_regularizer=l2(l2_reg),
                            attention=attention,
                            attn_kernel_regularizer=l2(l2_reg))([dropout2, A_in])

    # Build model
    optimizer = K.optimizers.Adam(lr=learning_rate)

    if original:
        model = K.models.Model(inputs=[X_in, A_in], outputs=graph_attention_2)
        model.compile(optimizer=optimizer,
                      loss="categorical_crossentropy",
                      weighted_metrics=["acc"])
    else:
        output = K.layers.Lambda(
                    lambda x: tf.reshape(tf.batch_gather(x, I_in),
                                         (-1, node_size, n_classes)))(graph_attention_2)
        model = K.models.Model(inputs=[X_in, A_in, I_in], outputs=output)
        model.compile(optimizer=optimizer,
                      loss="categorical_crossentropy",
                      metrics=["acc"])

    model.summary()

    # Callbacks
    experiment_dir = "log/gan_experiment"
    monitor = "val_acc"
    if original:
        experiment_dir += "_o"
        monitor = "val_weighted_acc"
    if not attention:
        experiment_dir += "_na"

    experiment_dir = storage.data_path(experiment_dir)
    model_path = os.path.join(experiment_dir, "best_model.h5")
    es_callback = K.callbacks.EarlyStopping(
                    monitor=monitor, patience=es_patience)
    tb_callback = K.callbacks.TensorBoard(log_dir=experiment_dir)
    mc_callback = K.callbacks.ModelCheckpoint(
                                model_path,
                                monitor=monitor,
                                save_best_only=True,
                                save_weights_only=True)

    def batch_generator(indices, label):
        if len(indices) != len(label):
            raise Exception("Does not match length")
        batch_size = len(indices)
        batch_size = batch_size // node_size

        def generator():
            while True:
                for i in range(batch_size):
                    _X = np.array([X])
                    _A = np.array([A])
                    samples = np.random.randint(len(indices), size=node_size)
                    _i = np.array([indices[samples]])
                    _label = np.array([label[samples]])
                    yield [_X, _A, _i], _label
        return generator(), batch_size

    if original:
        validation_data = ([X, A], Y_val, idx_val)
        model.fit([X, A],
                  Y_train,
                  sample_weight=idx_train,
                  epochs=epochs,
                  batch_size=N,
                  validation_data=validation_data,
                  shuffle=False,  # Shuffling data means shuffling the whole graph
                  callbacks=[es_callback, tb_callback, mc_callback])

        # Load best model
        model.load_weights(model_path)

        # Evaluate model
        eval_results = model.evaluate([X, A],
                                      Y_test,
                                      sample_weight=idx_test,
                                      batch_size=N,
                                      verbose=0)
    else:
        val_generator, val_steps = batch_generator(idx_val, Y_val)
        train_generator, train_steps = batch_generator(idx_train, Y_train)

        model.fit_generator(
                train_generator, train_steps,
                validation_data=val_generator, validation_steps=val_steps,
                epochs=epochs,
                callbacks=[es_callback, tb_callback, mc_callback])

        # Load best model
        model.load_weights(model_path)

        # Evaluate model
        test_generator, test_steps = batch_generator(idx_test, Y_test)
        eval_results = model.evaluate_generator(
                            test_generator, test_steps,
                            verbose=0)

    print("Done.\n"
          "Test loss: {}\n"
          "Test accuracy: {}".format(*eval_results))
