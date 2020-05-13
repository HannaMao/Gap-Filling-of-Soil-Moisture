# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause

from .model import MLModel
from ..metrics import pearson_corr_as_scorer
from ..train_test_split import AdaptiveKFold
from .utils import pre_process

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from scipy.stats import randint
from scipy.stats import uniform
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


def rmse(y, y_pred):
    return np.sqrt(np.mean((y_pred - y)**2))


class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    if epoch % 100 == 0: print('')
    print('.', end='')


def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
             label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
             label='Val loss')
    plt.legend()
    plt.show()


class NeuralNetworksModel(MLModel):
    def __init__(self, X_train, y_train, X_test, y_test, seed=192, param_dist=None):
        super(NeuralNetworksModel, self).__init__(X_train, y_train, X_test, y_test)
        self.param_dist = param_dist
        self.seed = seed

    def build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(200, activation=tf.nn.relu, input_shape=(self.X_train.shape[1],)),
            keras.layers.Dense(100, activation=tf.nn.relu),
            keras.layers.Dense(1)
        ])

        optimizer = tf.train.RMSPropOptimizer(0.001)

        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae'])
        return model

    def apply_model(self, feature_names, cv_type=None, search_type=None, scorer=None):
        self.cv_type = cv_type
        self.search_type = search_type
        self.scorer = scorer

        self.X_train, self.X_test = pre_process(self.X_train, self.X_test)
        if cv_type is None:
            nn = self.build_model()
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
            history = nn.fit(self.X_train, self.y_train, epochs=500, validation_split=0.2, verbose=0,
                             callbacks=[early_stop, PrintDot()])
            plot_history(history)

        oob_prediction, test_prediction = self.apply_predict(nn, oob=True)
        return oob_prediction, test_prediction
