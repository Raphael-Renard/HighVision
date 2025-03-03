import numpy as np
import cv2
import os

def add_gaussian_noise(image, mean=0, stddev=25):
    """
    Adds Gaussian noise to a grayscale or RGB image.

    Args:
        image (numpy.ndarray): Input image (grayscale or RGB).
        mean (float): Mean of the Gaussian distribution.
        stddev (float): Standard deviation of the Gaussian distribution.

    Returns:
        numpy.ndarray: Noisy image with the same dimensions as the input.
    """

    # Ensure the input image is a NumPy array
    image = image.astype(np.float32)

    # Generate Gaussian noise
    noise = np.random.normal(mean, stddev, image.shape)

    # Add the noise to the image
    noisy_image = image + noise

    # Clip the pixel values to stay in the valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)


import torch.nn as nn
class transforms_add_gaussian_noise(nn.Module):
    def __init__(self, mean=0, stddev=25):
        super(transforms_add_gaussian_noise, self).__init__()
        self.mean = mean
        self.stddev=stddev

    def __call__(self, batch):
        for image in batch:
            image = add_gaussian_noise(image, self.mean, self.stddev)
        return batch