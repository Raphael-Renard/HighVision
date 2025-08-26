import numpy as np
import sys
import glob
import cv2
import random
import os
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from absolute_path import absolutePath
from degradations.methods.utils import remove_black_borders, restore_black_borders


def random_partition(total, n_parts, min_frac=0.25, max_frac=0.75):
    """Découpe `total` en `n_parts` morceaux aléatoires avec contraintes sur la taille relative."""
    min_val = int(total * min_frac)
    max_val = int(total * max_frac)
    parts = []

    for i in range(n_parts - 1):
        remaining = total - sum(parts)
        max_possible = min(max_val, remaining - (n_parts - len(parts) - 1) * min_val)
        min_possible = min_val
        if max_possible < min_possible:
            # Repartir si impossible de respecter les contraintes
            return random_partition(total, n_parts, min_frac, max_frac)
        part = random.randint(min_possible, max_possible)
        parts.append(part)

    parts.append(total - sum(parts))
    return parts



def patchwork(image, max_ratio=0.25, contour_thickness=33):
    original_h, original_w = image.shape[:2]

    image = remove_black_borders(image)
    h, w = image.shape[:2]

    # 1. Marges aléatoires pour chaque bord
    min_ratio = 1/8
    top = random.randint(int(h * min_ratio), int(h * max_ratio))
    bottom = random.randint(int(h * min_ratio), int(h * max_ratio))
    left = random.randint(int(w * min_ratio), int(w * max_ratio))
    right = random.randint(int(w * min_ratio), int(w * max_ratio))

    # plus de chance de ne pas avoir de bord sur certains côtés
    configs_proba = [0.25, 0.2, 0.2, 0.05, 0.3]
    configs_tuple = {0: [1,1,1,1], # photo avec 0 bord collé au cadre
                     1: [1,1,1,0], # photo avec 1 bord collé au cadre
                     2: [1,1,0,0], # photo avec 2 bords collés au cadre
                     3: [1,0,1,0], # ...
                     4: [1,0,0,0]}
    config = np.roll(configs_tuple[np.random.choice(list(range(0,5)), p=configs_proba)], np.random.randint(0,4))
    top *= config[0]
    bottom *= config[1]
    left *= config[2]
    right*= config[3]

    new_h = h + top + bottom
    new_w = w + left + right

    # Créer une nouvelle image blanche
    image_etendue = np.full((new_h, new_w, 3), 255, dtype=np.uint8)

    # Coller l’image originale au centre
    image_etendue[top:top+h, left:left+w] = image

    # Définir les zones des marges
    zones = {
        "top":    ((0, 0), (top, new_w)),
        "bottom": ((new_h - bottom, 0), (new_h, new_w)),
        "left":   ((0, 0), (new_h, left)),
        "right":  ((0, new_w - right), (new_h, new_w))
    }

    # Charger toutes les images disponibles
    photo_files = glob.glob(absolutePath+"degradations/datasets/backgrounds/*")
    photo_files = [p.replace("\\", "/") for p in photo_files]

    # Pour chaque bord, découper en rectangles et coller des images
    for zone_name, ((y1, x1), (y2, x2)) in zones.items():
        zone_h = y2 - y1
        zone_w = x2 - x1

        if zone_h == 0 or zone_w == 0:
            continue

        n_rects = random.choices([1,2,3], weights=[0.5, 0.4, 0.1], k=1)[0]

        if zone_name in ["top", "bottom"]:
            widths = random_partition(zone_w, n_rects, min_frac=0.25, max_frac=0.75)
            rx1 = x1
            for w in widths:
                rx2 = rx1 + w
                insert_random_patch(
                    image_etendue,
                    (y1, rx1, y2, rx2),
                    photo_files,
                    contour_thickness
                )
                rx1 = rx2

        else:
            heights = random_partition(zone_h, n_rects, min_frac=0.25, max_frac=0.75)
            ry1 = y1
            for h in heights:
                ry2 = ry1 + h
                insert_random_patch(
                    image_etendue,
                    (ry1, x1, ry2, x2),
                    photo_files,
                    contour_thickness
                )
                ry1 = ry2

    return cv2.resize(image_etendue, (original_w,original_h))


def insert_random_patch(base_image, rect_coords, photo_files, contour_blanc):
    y1, x1, y2, x2 = rect_coords
    h, w = y2 - y1, x2 - x1

    # Zone utile (centrée dans le rectangle)
    patch_y1 = y1 
    patch_y2 = y2 
    patch_x1 = x1 
    patch_x2 = x2 

    patch_h = patch_y2 - patch_y1
    patch_w = patch_x2 - patch_x1

    # Tirer une image de fond
    path = random.choice(photo_files)
    fond = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if fond is None:
        return

    fond_h, fond_w = fond.shape[:2]
    origin_h, origin_w = base_image.shape[:2]
    if fond_h < patch_h or fond_w < patch_w:
        fond = cv2.resize(fond, (max(w, fond_w), max(h, fond_h)))
    
    elif fond_h > 2 * origin_h or fond_w > 2*origin_w:
        scale = random.uniform(1, 1.5)
        fond = cv2.resize(fond, (int(scale*origin_w), int(scale*origin_h)))

    # Zone aléatoire dans l'image de fond
    start_x = random.randint(0, fond.shape[1] - patch_w)
    start_y = random.randint(0, fond.shape[0] - patch_h)
    patch = fond[start_y:start_y + patch_h, start_x:start_x + patch_w]

    # Coller le patch dans l'image étendue
    base_image[patch_y1:patch_y2, patch_x1:patch_x2] = patch
    cv2.rectangle(base_image, (patch_x1, patch_y1), (patch_x2 , patch_y2), color=(255, 255, 255), thickness=contour_blanc)




class transforms_patchwork(nn.Module):
    def __init__(self, max_ratio=0.25, contour_thickness=2):
        super(transforms_patchwork, self).__init__()
        self.max_ratio = max_ratio
        self.contour_thickness = contour_thickness

    def __call__(self, batch):
        one_image = False
        if len(batch.shape) == 3: # si on ne passe qu'une image au lieu d'un batch
            batch = batch.unsqueeze(0)
            one_image = True
            
        results = torch.empty_like(batch)
        for i, image in enumerate(batch):
            image_array = np.transpose(np.array(image), (1, 2, 0)).copy() * 255
            image_array = image_array.astype(np.uint8)

            image = patchwork(image_array, self.max_ratio, self.contour_thickness)
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image)
            results[i] = image/255
        
        if one_image:
            results = results.squeeze(0)
        return results
    

