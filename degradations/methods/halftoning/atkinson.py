import numpy as np
import torch.nn as nn
import torch
import cv2

def atkinson_dithering(image, block_size=8):
    """
    Applies Atkinson dithering to a grayscale image with block processing.

    Args:
        image (numpy.ndarray): RGB input image.
        block_size (int): Size of the blocks for processing.

    Returns:
        numpy.ndarray: Dithered binary image.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

    dithered_image = cv2.cvtColor(dithered_image,cv2.COLOR_GRAY2RGB)
    return dithered_image



def atkinson_small(img, scale=5, block_size=8):
    """Agrandit l'image pour avoir une meilleure demi-teinte"""
    h, w,_ = img.shape

    new_size = (int(w * scale), int(h * scale))
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

    img = atkinson_dithering(img, block_size)

    h, w,_ = img.shape
    new_size = (int(w/scale), int(h/scale))
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
  
    return img


class transforms_atkinson_dithering(nn.Module):
    def __init__(self, block_size=8, scale=5):
        super(transforms_atkinson_dithering, self).__init__()
        self.block_size=block_size
        self.scale=scale

    def __call__(self, batch):
        results = torch.empty_like(batch)

        for i, image in enumerate(batch):
            image_array = np.array(image).swapaxes(0,2) * 255
            mask = np.where(image_array==0) # bords noirs
            
            if min(image_array.shape[0],image_array.shape[1])<1000:
                image = atkinson_small(image_array, self.scale, self.block_size)
            else:
                image = atkinson_dithering(image_array, self.block_size)

            image[mask] = 0
            image = np.array(image).swapaxes(0,2)
            image = torch.tensor(image)
            results[i] = image / 255
            
        return results
    
if __name__ =="__main__":
    image_path = "C:/Users/rapha/Documents/Cours/Master/Stage/HighVision/degradations/results/2K2476_16_01.jpg"
    #image_path = "C:/Users/rapha/Documents/Cours/Master/Stage/Data/Sena/FRAN_0568_11AR_699/FRAN_0568_000014_L.jpg"
    img = cv2.imread(image_path)
    img = atkinson_small(img)
    cv2.imwrite("atkinson.jpg",img)
