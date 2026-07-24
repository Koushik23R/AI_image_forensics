from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from tensorflow import keras

from .preprocessing import apply_fft_to_image, preprocess_image_for_spatial_branch


class CustomDataGenerator(keras.utils.Sequence):
    def __init__(self, image_paths, labels, img_size=(128, 128), batch_size=32, shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_image_paths = [self.image_paths[k] for k in indexes]
        batch_labels = [self.labels[k] for k in indexes]

        spatial_images = []
        fft_images = []
        valid_labels = []

        for img_path, label in zip(batch_image_paths, batch_labels):
            spatial_images.append(preprocess_image_for_spatial_branch(img_path, self.img_size))
            fft_images.append(apply_fft_to_image(img_path, self.img_size))
            valid_labels.append(label)

        spatial_images_batch = np.array(spatial_images)
        fft_images_batch = np.array(fft_images)
        labels_batch = np.array(valid_labels)

        return {"spatial_input": spatial_images_batch, "fft_input": fft_images_batch}, labels_batch

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)
