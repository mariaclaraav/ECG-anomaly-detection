import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, losses, Sequential
from tensorflow.keras.models import Model



class AnomalyDetector(Model):
    def __init__(self, input_dim):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(shape=(input_dim,)),  # Explicitly define the input shape
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu")
        ])
        
        self.decoder = tf.keras.Sequential([
             layers.Dense(32, activation="relu"),
             layers.Dense(16, activation="relu"),
            layers.Dense(input_dim, activation="sigmoid")  # Match the output shape to input shape
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded