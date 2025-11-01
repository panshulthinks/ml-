import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np

# Define paths to training and validation directories
train_dir = "dataset/training_set"
val_dir = "dataset/test_set"

# lets normalise our data
