import os
import sys
import numpy as np
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from build_model import build_model

def train_and_save_model(model_path='../models/mnist_model.keras', epochs=5):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    x_train = x_train[..., np.newaxis] / 255.0

    model = build_model()
    model.fit(x_train, y_train, epochs=epochs, validation_split=0.2)
    model.save(model_path)
    print(f"Model trained and saved as '{model_path}'.")
    return model

if __name__ == "__main__":
    train_and_save_model()
