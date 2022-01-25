import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

# logger
# tensorboard --logdir=./my_logs --port=6006
root_logdir = os.path.join(os.curdir, "my_logs")


def get_run_logdir():
    """
    returns a folder path for TensorBoard
    """
    import time

    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()

# load data
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train, y_train = X_train_full[:50000] / 255, y_train_full[:50000]
X_valid, y_valid = X_train_full[50000:] / 255, y_train_full[50000:]


def build_model(
    n_hidden=1,
    n_neurons=30,
    learning_rate=3e-3,
    input_shape=[28, 28],
    output_shape=10,
):
    """
    return build and compiled model with selected parameters
    """
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(output_shape, activation="softmax"))

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model


keras_clf = keras.wrappers.scikit_learn.KerasClassifier(build_model)

# callbacks
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    os.path.join(os.curdir, "models", "my_keras_model.h5")
)
early_stopping_cb = keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)

# Randomized Search
param_distribs = {
    "n_hidden": [1, 2, 3],
    "n_neurons": np.arange(1, 100),
    "learning_rate": reciprocal(3e-4, 3e-2),
}

rnd_search_cv = RandomizedSearchCV(keras_clf, param_distribs, n_iter=10, cv=3)

rnd_search_cv.fit(
    X_train,
    y_train,
    epochs=100,
    validation_data=(X_valid, y_valid),
    callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb],
)

print(rnd_search_cv.best_params_)
print(rnd_search_cv.best_score_)
