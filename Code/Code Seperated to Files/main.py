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
from DataUtils import *
from Model import UnetRegressor


if __name__ == "__main__":
    # fetch full paths to images
    paths = download_dataset_files('Dataset', '.jpg')
    paths = filter_and_print(paths)

    # split paths to train and test image files
    train_percentage = 0.8
    cutoff = int(train_percentage * len(paths))
    train_paths = paths[:cutoff]
    test_paths = paths[cutoff:]

    # download train and test datasets as tf dataset object
    train_dataset = download_dataset(train_paths)
    test_dataset = download_dataset(test_paths)

    model = UnetRegressor()
    model.fit(train_dataset, test_dataset, niter=30)
