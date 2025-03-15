import tensorflow as tf
from tensorflow.keras import layers, models

def build_model():
    data_augmentation = models.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1)
    ], name="data_augmentation")
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ], name="mnist_model")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
