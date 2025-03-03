import numpy as np
import cv2
import os
import torch.nn as nn
import torch

def atkinson_dithering(image, block_size=8):
    """
    Applies Atkinson dithering to a grayscale image with block processing.

    Args:
        image (numpy.ndarray): Grayscale input image.
        block_size (int): Size of the blocks for processing.

    Returns:
        numpy.ndarray: Dithered binary image.
    """
    image = image.astype(np.float32)
    height, width = image.shape
    dithered_image = np.zeros_like(image, dtype=np.uint8)

    # Process the image block-by-block
    for y_start in range(0, height, block_size):
        for x_start in range(0, width, block_size):
            # Define the block region
            block_end_y = min(y_start + block_size, height)
            block_end_x = min(x_start + block_size, width)
            block = image[y_start:block_end_y, x_start:block_end_x]

            # Apply Atkinson dithering within the block
            for y in range(block.shape[0]):
                for x in range(block.shape[1]):
                    old_pixel = block[y, x]
                    new_pixel = 255 if old_pixel > 127 else 0
                    dithered_image[y_start + y, x_start + x] = new_pixel
                    error = (old_pixel - new_pixel) / 8  # Distribute equally to 6 neighbors

                    # Distribute the error to neighboring pixels
                    if x + 1 < block.shape[1]:
                        block[y, x + 1] += error
                    if x + 2 < block.shape[1]:
                        block[y, x + 2] += error
                    if y + 1 < block.shape[0]:
                        if x > 0:
                            block[y + 1, x - 1] += error
                        block[y + 1, x] += error
                        if x + 1 < block.shape[1]:
                            block[y + 1, x + 1] += error
                    if y + 2 < block.shape[0]:
                        block[y + 2, x] += error

    return dithered_image



class transforms_atkinson_dithering(nn.Module):
    def __init__(self, block_size=8):
        super(transforms_atkinson_dithering, self).__init__()
        self.block_size=block_size

    def __call__(self, batch):
        for image in batch:
            image = atkinson_dithering(image, self.block_size)
        return batch.clamp(0, 1)