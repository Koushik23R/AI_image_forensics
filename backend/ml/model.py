from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_dual_branch_cnn(input_shape=(256, 256, 3)):
    """
    Builds a dual-branch CNN model for AI-generated image detection.
    One branch processes spatial (RGB) features, the other processes FFT (frequency) features.
    """
    spatial_input = keras.Input(shape=input_shape, name="spatial_input")
    x_spatial = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(spatial_input)
    x_spatial = layers.MaxPooling2D((2, 2))(x_spatial)
    x_spatial = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x_spatial)
    x_spatial = layers.MaxPooling2D((2, 2))(x_spatial)
    x_spatial = layers.Flatten()(x_spatial)
    x_spatial = layers.Dense(128, activation="relu")(x_spatial)

    fft_input = keras.Input(shape=(input_shape[0], input_shape[1], 1), name="fft_input")
    x_fft = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(fft_input)
    x_fft = layers.MaxPooling2D((2, 2))(x_fft)
    x_fft = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x_fft)
    x_fft = layers.MaxPooling2D((2, 2))(x_fft)
    x_fft = layers.Flatten()(x_fft)
    x_fft = layers.Dense(128, activation="relu")(x_fft)

    concatenated = layers.concatenate([x_spatial, x_fft])
    output = layers.Dense(1, activation="sigmoid")(concatenated)

    model = keras.Model(inputs=[spatial_input, fft_input], outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
