import numpy as np
import cv2
import torch.nn as nn
import torch

def generate_bayer_matrix(n):
    """
    Generates a Bayer matrix of size n x n.

    Args:
        n (int): The size of the Bayer matrix (must be a power of 2).

    Returns:
        numpy.ndarray: Generated Bayer matrix.
    """
    if n == 2:
        return np.array([[0, 2],
                         [3, 1]])
    smaller_matrix = generate_bayer_matrix(n // 2)
    upper_left = 4 * smaller_matrix
    upper_right = 4 * smaller_matrix + 2
    lower_left = 4 * smaller_matrix + 3
    lower_right = 4 * smaller_matrix + 1
    return np.block([[upper_left, upper_right],
                     [lower_left, lower_right]])

def bayer_halftoning(image, cell_size=4):
    """
    Applies basic halftoning using a Bayer threshold matrix.

    Args:
        image (numpy.ndarray): BGR input image.
        cell_size (int): Size of each halftoning cell.

    Returns:
        numpy.ndarray: Halftoned binary image.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float32)
    height, width = image.shape

    # Generate a Bayer matrix of the desired cell size
    bayer_matrix = generate_bayer_matrix(cell_size)

    # Scale Bayer matrix to 0–255 range
    bayer_matrix = (bayer_matrix / ((cell_size ** 2)-1)) * 255

    # Create an output image
    halftoned_image = np.zeros_like(image, dtype=np.uint8)

    # Apply the halftoning process
    for y in range(0, height, cell_size):
        for x in range(0, width, cell_size):
            # Extract the current cell from the image
            cell = image[y:y + cell_size, x:x + cell_size]

            # Compare each pixel in the cell with the Bayer matrix
            thresholded_cell = (cell > bayer_matrix[:cell.shape[0], :cell.shape[1]]) * 255

            # Place the thresholded cell in the output image
            halftoned_image[y:y + cell_size, x:x + cell_size] = thresholded_cell
    
    halftoned_image = cv2.cvtColor(halftoned_image,cv2.COLOR_GRAY2RGB)
    #halftoned_image = halftoned_image.repeat((3,1,1))
    return halftoned_image



class transforms_bayer_halftoning(nn.Module):
    def __init__(self, cell_size=4):
        super(transforms_bayer_halftoning, self).__init__()
        self.cell_size=cell_size

    def __call__(self, batch):
        results = torch.empty_like(batch)
        for i, image in enumerate(batch):
            image_array = np.array(image).swapaxes(0,2) * 255
            mask = np.where(image_array==0)
            image_half = bayer_halftoning(image_array, self.cell_size)
            image_half[mask] = 0
            image_half = image_half.swapaxes(0,2)
            results[i] = torch.tensor(image_half)/255
        return results