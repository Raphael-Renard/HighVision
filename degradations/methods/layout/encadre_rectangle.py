import cv2
import torch.nn as nn
import torch
import numpy as np
import random
import sys
import os
import glob
import math
import copy



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







def add_enluminures(masque, contour, couleur_type='blanc', motif_type='croix', thickness=3):
    """
    Dessine des motifs décoratifs autour de la zone masquée, en suivant le contour.
    """

    masque_result = masque.copy()
    h, w = masque.shape[:2]

    min_size = min(h, w)
    scale_radius = np.random.uniform(0.02, 0.04)
    motif_radius = int(min_size * scale_radius) 
    
    if couleur_type == 'blanc':
        color = random.randint(150, 255)
    elif couleur_type == 'noir':
        color = random.randint(0, 80)
    color = (color, color, color)
    
    if motif_type == 'random':
        motif_type = np.random.choice(['croix', 'étoile', 'triangle', 'point', 'fleur', 'bande'])

    for i, c in enumerate(contour):
        x,y = int(c[0]), int(c[1])

        if motif_type == 'croix':
            cv2.line(masque_result, (x - motif_radius, y), (x + motif_radius, y), color, thickness)
            cv2.line(masque_result, (x, y - motif_radius), (x, y + motif_radius), color, thickness)

        elif motif_type == 'point':
            cv2.circle(masque_result, (x, y), thickness*2, color, -1)

        elif motif_type == 'triangle':
            pts = []
            for a in [0, 120, 240]:
                a_rad = math.radians(a)
                x1 = int(x + motif_radius * math.cos(a_rad))
                y1 = int(y + motif_radius * math.sin(a_rad))
                pts.append((x1, y1))
            cv2.polylines(masque_result, [np.array(pts)], isClosed=True, color=color, thickness=thickness)

        elif motif_type == 'étoile':
            for a in range(0, 360, 45):
                x1 = int(x + motif_radius * math.cos(math.radians(a)))
                y1 = int(y + motif_radius * math.sin(math.radians(a)))
                cv2.line(masque_result, (x, y), (x1, y1), color, thickness)

        elif motif_type == 'fleur':
            petal_radius = motif_radius // 2
            petal_distance = motif_radius
            for a in range(0, 360, 60):
                a_rad = math.radians(a)
                x1 = int(x + petal_distance * math.cos(a_rad))
                y1 = int(y + petal_distance * math.sin(a_rad))
                cv2.circle(masque_result, (x1, y1), petal_radius, color, -1)
            cv2.circle(masque_result, (x, y), thickness//2, color, -1)
        
        elif motif_type == 'bande':
            for j in range(-1, 2):
                cv2.line(masque_result, (x - motif_radius, y + j*thickness), (x + motif_radius, y + j*thickness), color, thickness)

    return masque_result




def encadre_rectangle(image, couleur_fond='blanc', motif_type='random', 
                      contour_epaisseur = 33, bordures=200):
    """
    Applique un cadrage rond ou losange à une image.

    Args:
        image (np.array): Image d'entrée.
        couleur_fond (str): 'blanc' pour un fond blanc, 'noir' pour un fond noir.
        motif_type (str): Type de motif ('croix', 'étoile', 'triangle', 'point', 'fleur', 'random', 'bande').
        contour_epaisseur (int): Épaisseur du contour blanc rajouté autour de l'image (et éventuellement des enluminures).
        bordures (int): Taille de l'espace (en pixels) entre le bord de l'image et la forme.
    """

    h, w = image.shape[:2]

    # add bordure 
    if couleur_fond == 'blanc':
        color_value = (255, 255, 255)
        color_bande = (0, 0, 0)
    elif couleur_fond == 'noir':
        color_value = (0, 0, 0)
        color_bande = (255, 255, 255)
    image = cv2.copyMakeBorder(image, bordures, bordures, bordures, bordures, cv2.BORDER_CONSTANT, value=color_value)
    image = cv2.resize(image, (w, h))


    if motif_type == 'bande':
        nb_bandes = np.random.randint(1, 4) # nb_bandes rectangles imbriqués
        ecart = bordures // nb_bandes
        for i in range(1, nb_bandes+1):
            cv2.rectangle(image, (i*ecart, i*ecart), 
                          (w - i*ecart, h - i*ecart), 
                          color_bande, contour_epaisseur)
        return image



    else:
        nb_enluminures = np.random.randint(10, 40)
        coints = np.array([
            [bordures//2, bordures//2],          # haut
            [w-bordures//2, bordures//2],          # droite
            [w-bordures//2, h-bordures//2],          # bas
            [bordures//2, h-bordures//2]           # gauche
        ])
        total_length = (w-bordures + h-bordures)*2
        contour_enlu = []

        for i in range(-1, len(coints)-1):
            p1 = coints[i]
            p2 = coints[i+1]
            lenght = max(abs(p2[0]-p1[0]), abs(p2[1]-p1[1]))
            num_points = int(nb_enluminures * lenght/total_length)

            for j in range(num_points):
                t = j / num_points
                point = (1 - t) * p1 + t * p2
                contour_enlu.append(point)
   

 
    if couleur_fond == 'blanc':
        image = add_enluminures(image.copy(), contour_enlu, 'noir', motif_type=motif_type, thickness=contour_epaisseur)


    elif couleur_fond == 'noir':
        image = add_enluminures(image.copy(), contour_enlu, 'blanc', motif_type=motif_type, thickness=contour_epaisseur)
   
    else:
        raise ValueError("Couleur_fond non reconnue : choisir 'noir' ou 'blanc'")
    return image




class transforms_encadre_rectangle(nn.Module):
    def __init__(self, couleur_fond=None, motif_type='random', contour_epaisseur = 1, bordures=None):
        super(transforms_encadre_rectangle, self).__init__()
        self.couleur_fond = couleur_fond
        self.motif_type = motif_type
        self.contour_epaisseur = contour_epaisseur
        self.bordures = bordures

    def __call__(self, batch):
        one_image = False
        if len(batch.shape) == 3: # si on ne passe qu'une image au lieu d'un batch
            batch = batch.unsqueeze(0)
            one_image = True
            
        results = torch.empty_like(batch)
        for i, image in enumerate(batch):
            image_cp = copy.copy(image)
            image_array = np.transpose(np.array(image_cp), (1, 2, 0)) * 255
            image_array = image_array.astype(np.uint8)


            if self.couleur_fond is None:
                couleur_fond = random.choice(['blanc', 'noir'])
            else:
                couleur_fond = self.couleur_fond

            if self.bordures is None:
                bordures = random.randint(0, 10)
            else:
                bordures = self.bordures

            
            shape = image_array.shape
            image_array, black_borders = remove_black_borders(image_array)
            image = encadre_rectangle(image_array, couleur_fond, self.motif_type, self.contour_epaisseur, bordures)
            image = restore_black_borders(shape, image, black_borders)

            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image)
            results[i] = image/255
        
        if one_image:
            results = results.squeeze(0)
        return results
    