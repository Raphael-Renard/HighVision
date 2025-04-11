import cv2
import numpy as np
import torch.nn as nn
import torch
from noise import pnoise2
import glob

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
#from absolute_path import absolutePath
absolutePath = 'C:/Users/rapha/Documents/Cours/Master/Stage/HighVision/'


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



def apply_texture(cible):
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
    result = cv2.addWeighted(cible, 0.9, texture_resized, alpha, 0)
   
    return result.clip(0,255)




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
    crumpled_final= apply_texture(crumpled_final)

    # reconvert to RGB
    crumpled_final = cv2.cvtColor(crumpled_final,cv2.COLOR_GRAY2RGB)
    return crumpled_final.clip(0,255)




class transforms_crumpled_paper(nn.Module):
    def __init__(self, intensity_waves=10, intensity_blend=0.1):
        super(transforms_crumpled_paper, self).__init__()
        self.intensity_waves = intensity_waves
        self.intensity_blend = intensity_blend


    def __call__(self, batch):
        one_image = False
        if len(batch.shape) == 3: # si on ne passe qu'une image au lieu d'un batch
            batch = batch.unsqueeze(0)
            one_image = True

        results = torch.empty_like(batch)
        for i, image in enumerate(batch):
            image_array = np.array(image).swapaxes(0,2) * 255
            mask = np.where(image_array==0) # bords noirs
            image = crumpled_paper(image_array, intensity_waves=self.intensity_waves, 
                                   intensity_blend=self.intensity_blend)
            image[mask]=0
            image = np.array(image).swapaxes(0,2)
            image = torch.tensor(image) / 255
            results[i] = image
        
        if one_image:
            results = results.squeeze(0)
        return results








import cv2
import numpy as np
import torch
import torch.nn as nn


def remove_black_borders(image):
    """Supprime les bords noirs qui ont été rajoutés pour le dataloader."""
    # Créer un masque détectant les pixels non noirs
    mask = np.any(image > 0, axis=2)
    
    # Trouver les coordonnées des pixels non noirs
    coords = np.column_stack(np.where(mask))

    if len(coords) == 0:
        return image

    # Trouver la boîte englobante des pixels non noirs
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Recadre image
    cropped_image = image[y_min:y_max+1, x_min:x_max+1]

    return cropped_image, (y_min, x_min, y_max, x_max)  # image recadrée et coordonnées du recadrage

def restore_black_borders(original_shape, cropped_image, crop_coords):
    """Restaure les bords noirs après traitement, en gardant la taille originale."""
    h_original, w_original = original_shape[:2]
    h_cropped, w_cropped = cropped_image.shape[:2]

    # Créer une image noire avec la taille originale
    restored_image = np.zeros((h_original, w_original, 3), dtype=np.uint8)

    # Récupérer les anciennes coordonnées de la zone non noire
    y_min, x_min, _, _ = crop_coords

    # Coller l’image recadrée au bon emplacement
    restored_image[y_min:y_min+h_cropped, x_min:x_min+w_cropped] = cropped_image

    return restored_image



if __name__ =="__main__":
    image_path = "C:/Users/rapha/Documents/Cours/Master/Stage/HighVision/degradations/results/2K2476_16_01.jpg"
    #image_path = "C:/Users/rapha/Documents/Cours/Master/Stage/Data/Sena/FRAN_0568_11AR_699/FRAN_0568_000014_L.jpg"
    img = cv2.imread(image_path)

    shape = img.shape
    img, black_borders = remove_black_borders(img)
    img = crumpled_paper(img,10,0.3).astype(np.uint8)
    img = restore_black_borders(shape, img, black_borders)

    #img = crumpled_paper(img,50).astype(np.uint8)
    cv2.imwrite("crumpled.jpg",img)