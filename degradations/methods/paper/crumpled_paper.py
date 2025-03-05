import cv2
import numpy as np
import torch.nn as nn
import torch
from noise import pnoise2  # Perlin noise

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from absolute_path import absolutePath


#absolutePath = 'C:/Users/rapha/Documents/Cours/Master/Stage/HighVision/'

def generate_perlin_noise(shape, scale=200, octaves=3):
    """
    Generates Perlin noise for subtle crumpled paper texture.
    
    Parameters:
        shape (tuple): (height, width) of the noise map.
        scale (int): Controls wrinkle size (higher = larger wrinkles).
        octaves (int): Controls wrinkle complexity.
        
    Returns:
        numpy.ndarray: Perlin noise map (grayscale).
    """
    height, width = shape
    noise_img = np.zeros((height, width), np.float32)

    for i in range(height):
        for j in range(width):
            noise_img[i, j] = pnoise2(i / scale, j / scale, octaves=octaves)

    # Normalize noise to range 0-255 and enhance contrast
    noise_img = cv2.normalize(noise_img, None, 0, 255, cv2.NORM_MINMAX)
    return noise_img.astype(np.uint8)


import glob

def apply_texture(img):
    """Blends a folded paper texture onto an image."""
    texture_files = glob.glob(absolutePath+"degradations/datasets/crumpled_texture/*")
    texture_path = np.random.choice(texture_files)    
    texture_path = texture_path.replace("\\", "/")
    texture = cv2.imdecode(np.fromfile(texture_path, np.uint8), cv2.IMREAD_GRAYSCALE)

    # Resize texture to match image size
    texture = cv2.resize(texture, (img.shape[1], img.shape[0]))
 
    # Blend using multiply (darker paper effect)
    blended = cv2.addWeighted(img.astype(np.uint8),0.7,texture.astype(np.uint8),0.3,0)
    return blended

def crumpled_paper(img, intensity_waves=10, intensity_blend=0.1):
    """
    Applies a crumpled paper effect to an image.
    
    Parameters:
        img (numpy.ndarray): Input image (RGB).
        intensity_waves (int): Intensity of the distortion of the paper (lower = less crumpling).
        intensity_blend (float): Intensity of the texture added to the image
        
    Returns:
        numpy.ndarray: Grayscale image with crumpled paper effect.
    """
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = gray_img.shape[:2]

    # Generate Perlin noise with larger wrinkles
    noise_map = generate_perlin_noise((rows, cols), scale=400, octaves=5)

    # Create displacement fields based on the noise
    displacement_x = (noise_map.astype(np.float32) / 255.0 - 0.5) * intensity_waves
    displacement_y = (noise_map.astype(np.float32) / 255.0 - 0.5) * intensity_waves

    # Create mesh grid for pixel remapping
    map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
    map_x = (map_x + displacement_x).astype(np.float32)
    map_y = (map_y + displacement_y).astype(np.float32)

    # Apply subtle displacement
    crumpled_gray = cv2.remap(gray_img, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    # Blend original with Perlin noise
    crumpled_final = cv2.addWeighted(crumpled_gray, (1-intensity_blend), noise_map.astype(np.float32), intensity_blend, 0)
    crumpled_final= apply_texture(crumpled_final)

    # reconvert to RGB
    crumpled_final = cv2.cvtColor(crumpled_final,cv2.COLOR_GRAY2RGB)
    #crumpled_final = crumpled_final.repeat((3,1,1))
    return crumpled_final



class transforms_crumpled_paper(nn.Module):
    def __init__(self, intensity_waves=10, intensity_blend=0.1):
        super(transforms_crumpled_paper, self).__init__()
        self.intensity_waves = intensity_waves
        self.intensity_blend = intensity_blend


    def __call__(self, batch):
        results = torch.empty_like(batch)
        for i, image in enumerate(batch):
            image_array = np.array(image).swapaxes(0,2) * 255
            mask = np.where(image_array==0)
            image = crumpled_paper(image_array, intensity_waves=self.intensity_waves, 
                                   intensity_blend=self.intensity_blend)
            image[mask]=0
            image = np.array(image).swapaxes(0,2)
            image = torch.tensor(image) / 255
            results[i] = image
        return results



if __name__ == "__main__":
    # Example usage
    path = "degradations/datasets/original/"
    img = cv2.imread(path+"FRAN_0568_000183_L.jpg")
    crumpled_img = crumpled_paper(img)

    cv2.imshow("Crumpled Paper Effect", crumpled_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
