import cv2
import numpy as np
import torch.nn as nn
import torch
from noise import pnoise2  # Perlin noise
import glob

import sys
import os
"""
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from absolute_path import absolutePath
"""
absolutePath = 'C:/Users/rapha/Documents/Cours/Master/Stage/HighVision/'

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



def apply_texture2(cible):
    texture_files = glob.glob(absolutePath+"degradations/datasets/crumpled_texture/*")
    texture_path = np.random.choice(texture_files)    
    texture_path = texture_path.replace("\\", "/")
    texture = cv2.imdecode(np.fromfile(texture_path, np.uint8), cv2.IMREAD_GRAYSCALE)

    # Transformer en spectre de fréquence (FFT)
    dft = cv2.dft(np.float32(texture), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Supprimer les basses fréquences pour isoler la texture
    rows, cols = texture.shape
    mask = np.ones((rows, cols, 2), np.uint8)
    mask[rows//2-30:rows//2+30, cols//2-30:cols//2+30] = 0
    dft_shift = dft_shift * mask

    # Revenir à l'espace temporel
    f_ishift = np.fft.ifftshift(dft_shift)
    texture_filtered = cv2.idft(f_ishift)
    texture_filtered = cv2.magnitude(texture_filtered[:, :, 0], texture_filtered[:, :, 1])

    # Normaliser la texture filtrée
    texture_filtered = cv2.normalize(texture_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Redimensionner pour correspondre à l'image cible
    texture_resized = cv2.resize(texture_filtered, (cible.shape[1], cible.shape[0]))
    mean_texture = texture_filtered.mean()
    adjustment = 128 - mean_texture  # Pour équilibrer autour de la moitié de l'échelle des gris
    texture_filtered = np.clip(texture_filtered + adjustment, 0, 255).astype(np.uint8)

    # Mélanger les images
    alpha = 0.5  # Intensité de la texture
    result = cv2.addWeighted(cible, 0.8, texture_resized, alpha, 0)
   
    return result.clip(0,255)



def apply_texture(img):
    """Blends a folded paper texture onto an image."""
    texture_files = glob.glob(absolutePath+"degradations/datasets/crumpled_texture/*")
    texture_path = np.random.choice(texture_files)    
    texture_path = texture_path.replace("\\", "/")
    texture = cv2.imdecode(np.fromfile(texture_path, np.uint8), cv2.IMREAD_GRAYSCALE)

    # Resize texture to match image size
    texture = (255 - cv2.resize(texture, (img.shape[1], img.shape[0])))*2

    # Blend using multiply
    #blended = cv2.multiply(img.astype(np.float32) / 255.0, texture.astype(np.float32) / 255.0) *255
    blended = cv2.addWeighted(img.astype(np.uint8),0.8,texture.astype(np.uint8),0.2,0)
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
    crumpled_final = cv2.addWeighted(crumpled_gray.astype(np.uint8), (1-intensity_blend), noise_map.astype(np.uint8), intensity_blend, 0)
    crumpled_final= apply_texture2(crumpled_final)

    # reconvert to RGB
    crumpled_final = cv2.cvtColor(crumpled_final,cv2.COLOR_GRAY2RGB)
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



if __name__ =="__main__":
    image_path = "C:/Users/rapha/Documents/Cours/Master/Stage/HighVision/degradations/results/2K2476_16_01.jpg"
    #image_path = "C:/Users/rapha/Documents/Cours/Master/Stage/Data/Sena/FRAN_0568_11AR_699/FRAN_0568_000014_L.jpg"
    img = cv2.imread(image_path)
    img = crumpled_paper(img).astype(np.uint8)
    cv2.imshow("crumpled",img)
    cv2.waitKey(0)
    #cv2.imwrite("crumpled.jpg",img)