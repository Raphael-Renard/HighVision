import numpy as np
import torch.nn as nn
import torch
import cv2

def add_gaussian_noise(image, mean=0, stddev=25):
    """
    Adds Gaussian noise to a RGB image.

    Args:
        image (numpy.ndarray): Input image (RGB).
        mean (float): Mean of the Gaussian distribution.
        stddev (float): Standard deviation of the Gaussian distribution.

    Returns:
        numpy.ndarray: Noisy image with the same dimensions as the input.
    """

    # Ensure the input image is a NumPy array
    image = image.astype(np.float32)

    # Generate Gaussian noise
    noise = np.random.normal(mean, stddev, image.shape[:2])
    grayscale_noise = np.stack([noise]*3, axis=-1)

    # Add the noise to the image
    noisy_image = image + grayscale_noise

    # Clip the pixel values to stay in the valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)



class transforms_add_gaussian_noise(nn.Module):
    def __init__(self, mean=0, stddev=10):
        super(transforms_add_gaussian_noise, self).__init__()
        self.mean = mean
        self.stddev=stddev

    def __call__(self, batch):
        one_image = False
        if len(batch.shape) == 3: # si on ne passe qu'une image au lieu d'un batch
            batch = batch.unsqueeze(0)
            one_image = True
            
        results = torch.empty_like(batch)
        for i, image in enumerate(batch):
            image_array = np.array(image).swapaxes(0,2) * 255
            mask = np.where(image_array==0)
            image = add_gaussian_noise(image_array, self.mean, self.stddev)
            image[mask]=0
            image = torch.tensor(image).swapaxes(0,2)
            results[i] = image / 255
        
        if one_image:
            results = results.squeeze(0)
        return results
    

if __name__ =="__main__":
    image_path = "C:/Users/rapha/Documents/Cours/Master/Stage/HighVision/degradations/results/2K2476_16_01.jpg"
    #image_path = "C:/Users/rapha/Documents/Cours/Master/Stage/Data/Sena/FRAN_0568_11AR_699/FRAN_0568_000014_L.jpg"
    img = cv2.imread(image_path)
    img = add_gaussian_noise(img,stddev=10)
    cv2.imwrite("gaussian_noise.jpg",img)