import numpy as np
import torch.nn as nn
import torch
import cv2

def add_salt_and_pepper_noise(image, probability=0.01):
    """
    Adds salt and pepper noise to a grayscale or RGB image.

    Args:
        image (numpy.ndarray): Input image (grayscale or RGB).
        probability (float): Probability of a pixel being altered by noise (value between 0 and 1).

    Returns:
        numpy.ndarray: Noisy image with the same dimensions as the input.
    """

    # Ensure the input image is a NumPy array
    image = image.astype(np.uint8)
    noisy_image = np.copy(image)

    # Generate a random matrix the same size as the image
    random_matrix = np.random.rand(*image.shape[:2])

    # Salt (white pixels)
    noisy_image[random_matrix < (probability / 2)] = 255

    # Pepper (black pixels)
    noisy_image[random_matrix > (1 - probability / 2)] = 0
    return noisy_image


class transforms_add_salt_and_pepper_noise(nn.Module):
    def __init__(self, probability=0.01):
        super(transforms_add_salt_and_pepper_noise, self).__init__()
        self.probability = probability

    def __call__(self, batch):
        results = torch.empty_like(batch)
        for i, image in enumerate(batch):
            image_array = np.array(image).swapaxes(0,2) * 255
            mask = np.where(image_array==0)
            image = add_salt_and_pepper_noise(image_array, self.probability)
            image[mask]=0
            image = np.array(image).swapaxes(0,2)
            image = torch.tensor(image)
            results[i] = image / 255
        return results
    
if __name__ =="__main__":
    image_path = "C:/Users/rapha/Documents/Cours/Master/Stage/HighVision/degradations/results/2K2476_16_01.jpg"
    #image_path = "C:/Users/rapha/Documents/Cours/Master/Stage/Data/Sena/FRAN_0568_11AR_699/FRAN_0568_000014_L.jpg"
    img = cv2.imread(image_path)
    img = add_salt_and_pepper_noise(img)
    cv2.imwrite("salt_and_pepper.jpg",img)