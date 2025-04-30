import cv2
import numpy as np
import torch
import torch.nn as nn
import random
from skimage.segmentation import slic


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




def paint_with_slic(img, n_segments=200, brightness_boost=20, paint_ratio=0.1,
                    min_size_factor=0.1):

    h, w, _ = img.shape
    scale = 0.1
    #img = cv2.resize(img, (int(w*scale), int(h*scale)))  # redimension pour traitement
    

    # Convertir l’image pour SLIC
    segments = slic(img, n_segments=n_segments, compactness=5,  start_label=1, min_size_factor=min_size_factor)


    # Nombre de zones à peindre
    unique_segments = np.unique(segments)
    selected_labels = np.random.choice(unique_segments, size=int(len(unique_segments) * paint_ratio), replace=False)

    # Peinture par superpixel
    painted_layer = np.zeros_like(img)
    for label in selected_labels:
        mask = segments == label
        if np.sum(mask) < 300: continue  # ignorer petites zones
        region = img[mask]
        mean_color = np.mean(region, axis=0)
        light_color = np.clip(mean_color + brightness_boost, 0, 255).astype(np.uint8)
        painted_layer[mask] = light_color

    # Fusion avec image originale
    overlay = cv2.addWeighted(img, 1, painted_layer, 1, 0)
    return overlay







class transforms_paint(nn.Module):
    def __init__(self, n_segments=(100,200), brightness_boost=(10,30), paint_ratio=(0.1,0.3)):
        super(transforms_paint, self).__init__()
        self.n_segments = n_segments
        self.brightness_boost = brightness_boost
        self.paint_ratio = paint_ratio

    def __call__(self, batch):
        one_image = False
        if len(batch.shape) == 3: # si on ne passe qu'une image au lieu d'un batch
            batch = batch.unsqueeze(0)
            one_image = True
            
        results = torch.empty_like(batch)
        for i, image in enumerate(batch):
            image_array = np.transpose(np.array(image), (1, 2, 0)).copy() * 255
            image_array = image_array.astype(np.uint8)

            n_segments = random.randint(self.n_segments[0], self.n_segments[1])
            brightness_boost = random.randint(self.brightness_boost[0], self.brightness_boost[1])
            paint_ratio = random.uniform(self.paint_ratio[0], self.paint_ratio[1])

            image = paint_with_slic(image_array, n_segments, brightness_boost, paint_ratio)
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image)
            results[i] = image/255
        
        if one_image:
            results = results.squeeze(0)
        return results



