import numpy as np
import torch.nn as nn
import torch
import cv2
import random



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







def generate_border_polygon(w, h, intensity, side, num_points):
        """Génère un polygone fermé pour effacer un bord."""
        points = []
        border_points = []

        if side == "top":
            for i in np.linspace(0, w, num_points, dtype=int):
                y = np.random.randint(0, intensity)  # Hauteur aléatoire
                points.append((i, y))
            border_points = [(w, 0), (0, 0)]  # Bord réel
        elif side == "bottom":
            for i in np.linspace(0, w, num_points, dtype=int):
                y = h - np.random.randint(0, intensity)
                points.append((i, y))
            border_points = [(w, h), (0, h)]
        elif side == "left":
            for i in np.linspace(0, h, num_points, dtype=int):
                x = np.random.randint(0, intensity)
                points.append((x, i))
            border_points = [(0, h), (0, 0)]
        elif side == "right":
            for i in np.linspace(0, h, num_points, dtype=int):
                x = w - np.random.randint(0, intensity)
                points.append((x, i))
            border_points = [(w, h), (w, 0)]
        # Créer le polygone fermé
        polygon = np.array(points + border_points, np.int32)
        return polygon


def add_torn_corner(w, h, mask, position, max_offset = 40):
        """Ajoute un triangle irrégulier noir à un coin donné."""

        if position == "tl":  # Top Left
            points = np.array([[0, 0], 
                               [0,random.randint(10, max_offset)],
                               [random.randint(10, max_offset//2), random.randint(10, max_offset//2)], 
                               [random.randint(10, max_offset), 0]], np.int32)
        elif position == "tr":  # Top Right
            points = np.array([[w, 0], 
                               [w, random.randint(10, max_offset)],
                               [w - random.randint(10, max_offset//2), random.randint(10, max_offset//2)], 
                               [w - random.randint(10, max_offset), 0]], np.int32)
        elif position == "bl":  # Bottom Left
            points = np.array([[0, h], 
                               [0, h-random.randint(10, max_offset)],
                               [random.randint(10, max_offset//2), h - random.randint(10, max_offset//2)], 
                               [random.randint(10, max_offset), h]], np.int32)
        elif position == "br":  # Bottom Right
            points = np.array([[w, h], 
                               [w, h - random.randint(10, max_offset)],
                               [w - random.randint(10, max_offset//2), h - random.randint(10, max_offset//2)], 
                               [w - random.randint(10, max_offset), h]], np.int32)
        
        cv2.fillPoly(mask, [points], 0)  # Remplir le coin en noir


def torn_paper(image, intensity=10, corner_chance=.5, max_offset=40):
    """
    Simule des bords et coins déchirés sur une image.

    image (array): Image d'entrée (grayscale ou RGB).
    intensity (int): Degré d'irrégularité des bords.
    max_offset (int): Taille max des coins arrachés
    """
    h, w = image.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255

    sides = ["top", "bottom", "left", "right"]
    for side in sides:
        num_points = np.random.randint(10,15)
        polygon = generate_border_polygon(w,h,intensity,side,num_points)
        cv2.fillPoly(mask, [polygon], 0)

    # Appliquer le masque à l'image
    torn_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Ajouter des coins déchirés aléatoirement
    corners = ["tl", "tr", "bl", "br"]
    selected_corners = random.sample(corners, k=random.randint(1, 3))  # Sélectionne 1 à 3 coins au hasard
    for corner in selected_corners:
        if random.random() < corner_chance:
            add_torn_corner(w, h, mask, corner, max_offset)

    # Appliquer le masque à l'image
    torn_image = cv2.bitwise_and(image, image, mask=mask)
    return torn_image


class transforms_torn_paper(nn.Module):
    def __init__(self, intensity=10, corner_chance=.5, max_offset=40):
        super(transforms_torn_paper, self).__init__()
        self.intensity = intensity
        self.corner_chance = corner_chance
        self.max_offset = max_offset

    def __call__(self, batch):
        one_image = False
        if len(batch.shape) == 3: # si on ne passe qu'une image au lieu d'un batch
            batch = batch.unsqueeze(0)
            one_image = True

        results = torch.empty_like(batch)
        for i, image in enumerate(batch):
            image_array = np.array(image).swapaxes(0,2) * 255
            shape = image_array.shape
            img, black_borders = remove_black_borders(image_array)

            img = torn_paper(img, self.intensity,
                                 self.corner_chance, self.max_offset)

            img = restore_black_borders(shape, img, black_borders)
            image = np.array(img).swapaxes(0,2)
            image = torch.tensor(image)
            results[i] = image / 255
        
        if one_image:
            results = results.squeeze(0)
        return results
    
