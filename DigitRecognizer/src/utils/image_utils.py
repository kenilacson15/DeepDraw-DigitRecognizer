import numpy as np
from PIL import Image, ImageFilter
import tensorflow as tf

def preprocess_image(image, apply_smoothing=False, apply_normalization=True):
    if apply_smoothing:
        image = image.filter(ImageFilter.SMOOTH)
    image = image.resize((28, 28)).convert('L')
    img_array = np.array(image, dtype='float32')
    if apply_normalization:
        img_array /= 255.0
    return img_array.reshape(1, 28, 28, 1)

def apply_augmentation(img_array, augmentation_layer):
    img_tensor = tf.convert_to_tensor(img_array)
    augmented = augmentation_layer(img_tensor, training=True)
    return augmented.numpy()

def generate_heatmap(model, img_array, last_conv_layer=None):
    if last_conv_layer is None:
        last_conv_layer = next(
            (layer for layer in reversed(model.layers) if isinstance(layer, tf.keras.layers.Conv2D)),
            None
        )
        if last_conv_layer is None:
            raise ValueError("No convolutional layer found in the model")
    grad_model = tf.keras.models.Model(
        inputs=model.inputs, outputs=[last_conv_layer.output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    conv_outputs *= pooled_grads.reshape(1, 1, -1)
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)
    return heatmap
