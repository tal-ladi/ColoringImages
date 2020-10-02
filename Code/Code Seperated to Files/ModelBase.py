# Model class Definitions
# Imports
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, Add, Conv2DTranspose, MaxPooling2D, concatenate, Flatten, Dense
from PIL import Image
from google.colab import files

# Constants
image_size = (256, 256)  # power of two recommended for downsampling
batch_size = 1  # SGD


class ImageColorizerBase(object):
    def __init__(self, model):
        self.model = model
        self.history = None

    def save_results(self, name):
        self.weights_file = name + '.h5'
        self.history_file = name + '_history'

        try:
            self.model.save(self.weights_file)
        except:
            pass

        try:
            with open(self.history_file, 'wb') as file:
                pickle.dump(self.history.history, file)
        except:
            pass

        try:
            files.download(self.weights_file)
        except:
            pass

        try:
            files.download(self.history_file)
        except:
            pass

    def fit(self, train, validation, niter=1):

        physical_devices = tf.config.list_physical_devices('GPU')

        if len(physical_devices) > 0:
            device = "/GPU:0"
        else:
            device = "/CPU:0"

        with tf.device(device):
            self.history = self.model.fit(train,
                                          validation_data=validation,
                                          epochs=niter)
