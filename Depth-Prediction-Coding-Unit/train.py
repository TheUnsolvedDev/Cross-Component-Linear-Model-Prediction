import os
import sys
import random
import math
from wsgiref import validate
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import *
from dataset import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class Train:
    def __init__(self, data) -> None:
        self.data_obj = data
        self.train = self.data_obj.train
        self.test = self.data_obj.test
        self.validation = self.data_obj.validation
        self.callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True),
            tf.keras.callbacks.ModelCheckpoint('./model.h5', save_weights_only=True, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, verbose=1),
        ]

    def train_fit(self, epochs=2000, verbose=1):
        model = Model()
        model.summary()
        history = model.fit(self.train, epochs=epochs, verbose=verbose,
                            validation_data=self.validation, callbacks=self.callbacks)
        return history.history


if __name__ == '__main__':
    data = read_data_sets()
    train = data.train
    test = data.test
    validation = data.validation

    obj = Train(data)
    obj.train_fit()
