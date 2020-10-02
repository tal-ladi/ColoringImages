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

# Constants
image_size = (256, 256)  # power of two recommended for downsampling
batch_size = 1  # SGD


# Dataset Function Definitions

def download_dataset_files(folderpath: str, ext: str):
    """
    filepath: str:: full or incomplete path to dataset folder.
    ext: desired extention, for images it probably is .jpg or .png.
    """
    image_files = []
    for folder, _, filenames in os.walk(folderpath):
        for filename in filenames:
            full_path = os.path.join(folder, filename)
            if full_path.endswith(ext):
                image_files.append(full_path)
    return np.array(image_files)


def filter_and_print(paths: str):
    """
    Check if path is infact an image that can be opened
    into python.
    """
    valid_paths = []
    for path in paths:
        try:
            image = Image.open(path)
            image.verify()
            valid_paths.append(path)
        except Exception as e:
            print("couldn't open file {}, got error {}".format(path, e))
    return valid_paths


def read_image_tf(file):
    output = tf.io.read_file(file)
    output = tf.image.decode_jpeg(output, channels=3)
    output = tf.image.resize(output, image_size)
    return output


def rgb_to_grayscale(image):
    grayscale_image = tf.image.rgb_to_grayscale(image)
    grayscale_image_normalized = tf.math.divide(grayscale_image, 255)
    return grayscale_image_normalized


def download_dataset(paths):
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    rgb_images = dataset.map(read_image_tf)
    grayscale_images = rgb_images.map(rgb_to_grayscale)
    dataset = tf.data.Dataset.zip((grayscale_images, rgb_images))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)  # optimize pre-fetching off data to speed up computation
    return dataset
