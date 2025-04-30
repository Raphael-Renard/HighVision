import cv2
import torch.nn as nn
import numpy as np
import torch



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



def contraste(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    luminosite_moy = np.mean(img)
    
    if luminosite_moy<100:
        img = np.where(img < luminosite_moy/3, 0, img) # noir
        img = np.where(img > 2*luminosite_moy/3, img*1.5, img) # blanc
        img = np.where(img < luminosite_moy/2, img*0.2, img) # gris
    else:   
        img = np.where(img < luminosite_moy/3, 0, img) # noir
        img = np.where(img > 2*luminosite_moy/3, img*1.5, img) # blanc
        img = np.where(img < luminosite_moy/2, img*0.1, img) # gris
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    



class transforms_contrast(nn.Module):
    def __init__(self):
        super(transforms_contrast, self).__init__()

    def __call__(self, batch):
        one_image = False
        if len(batch.shape) == 3: # si on ne passe qu'une image au lieu d'un batch
            batch = batch.unsqueeze(0)
            one_image = True

        results = torch.empty_like(batch)
        for i, image in enumerate(batch):
            image_array = np.array(image).swapaxes(0,2) * 255
            shape = image_array.shape
            image_array, black_borders = remove_black_borders(image_array)
            image = contraste(image_array)
            image = restore_black_borders(shape, image, black_borders)
            image = np.array(image).swapaxes(0,2)
            image = torch.tensor(image)
            results[i] = image / 255
        
        if one_image:
            results = results.squeeze(0)
        return results
    

