import numpy as np
import cv2
import torch.nn as nn
import torch

def floyd_steinberg_halftoning(image, block_size=8):
    """
    Applies Floyd-Steinberg error diffusion halftoning with enhanced visibility for a newspaper-style effect.

    Args:
        image (numpy.ndarray): RGB input image.
        block_size (int): Size of the grid for coarser halftone visibility.

    Returns:
        numpy.ndarray: Halftoned binary image.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ensure the image is a NumPy array and in float format for processing
    image = image.astype(np.float32)
    height, width = image.shape

    # Create the output halftoned image
    halftoned_image = np.zeros_like(image, dtype=np.uint8)

    # Process the image in blocks
    for y_start in range(0, height, block_size):
        for x_start in range(0, width, block_size):
            # Define the block region
            block_end_y = min(y_start + block_size, height)
            block_end_x = min(x_start + block_size, width)
            block = image[y_start:block_end_y, x_start:block_end_x]

            # Apply Floyd-Steinberg error diffusion within the block
            for y in range(block.shape[0]):
                for x in range(block.shape[1]):
                    old_pixel = block[y, x]
                    new_pixel = 255 if old_pixel > 127 else 0
                    halftoned_image[y_start + y, x_start + x] = new_pixel
                    error = old_pixel - new_pixel

                    # Distribute error to neighboring pixels in the block
                    if x + 1 < block.shape[1]:  # Right
                        block[y, x + 1] += error * (7 / 16)
                    if y + 1 < block.shape[0]:  # Bottom
                        if x > 0:  # Bottom-left
                            block[y + 1, x - 1] += error * (3 / 16)
                        block[y + 1, x] += error * (5 / 16)  # Bottom-center
                        if x + 1 < block.shape[1]:  # Bottom-right
                            block[y + 1, x + 1] += error * (1 / 16)
    halftoned_image = cv2.cvtColor(halftoned_image,cv2.COLOR_GRAY2RGB)
    return halftoned_image


def floyd_small(img, scale=5, block_size=8):
    """Agrandit l'image pour avoir une meilleure demi-teinte"""
    h, w,_ = img.shape

    new_size = (int(w * scale), int(h * scale))
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

    img = floyd_steinberg_halftoning(img, block_size)

    h, w,_ = img.shape
    new_size = (int(w/scale), int(h/scale))
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
  
    return img


class transforms_floyd_steinberg_halftoning(nn.Module):
    def __init__(self, block_size=8, scale=5):
        super(transforms_floyd_steinberg_halftoning, self).__init__()
        self.block_size=block_size
        self.scale=scale

    def __call__(self, batch):
        results = torch.empty_like(batch)
        for i, image in enumerate(batch):

            image_array = np.array(image).swapaxes(0,2) * 255

            mask = np.where(image_array==0) # bords noirs

            if min(image_array.shape[0],image_array.shape[1])<1000:
                image = floyd_small(image_array, self.scale, self.block_size)
            else:
                image = floyd_steinberg_halftoning(image_array, self.block_size)

            image[mask] = 0

            image = np.array(image).swapaxes(0,2)

            image = torch.tensor(image)
            results[i] = image / 255
        return results
    

if __name__ =="__main__":
    image_path = "C:/Users/rapha/Documents/Cours/Master/Stage/HighVision/degradations/results/2K2476_16_01.jpg"
    #image_path = "C:/Users/rapha/Documents/Cours/Master/Stage/Data/Sena/FRAN_0568_11AR_699/FRAN_0568_000014_L.jpg"
    img = cv2.imread(image_path)

    h, w,_ = img.shape
    if min(h, w) < 300: # Redimensionner si trop petite
        img = floyd_small(img)
    else:
        img = floyd_steinberg_halftoning(img)
    
    cv2.imwrite("floyd.jpg",img)