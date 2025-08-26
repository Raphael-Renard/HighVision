import cv2
import torch.nn as nn
import torch
import numpy as np
import random
import sys
import os
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from absolute_path import absolutePath
from degradations.methods.utils import remove_black_borders, restore_black_borders




def add_formes_fond(fond, couleur_type, nombre_formes=30):
    h, w, _ = fond.shape

    for _ in range(nombre_formes):
        epaisseur = random.randint(1, 3)

        if couleur_type == 'blanc':
            grey = random.randint(150, 255)
            couleur = (grey, grey, grey)
        elif couleur_type == 'noir':
            grey = random.randint(0, 80)
            couleur = (grey, grey, grey)
        else:
            raise ValueError("Type de couleur non reconnu : choisir 'noir' ou 'blanc'.")

        forme = random.choice(['cercle', 'rectangle', 'ligne', 'triangle'])

        x1, y1 = random.randint(0, w), random.randint(0, h)
        x2, y2 = random.randint(0, w), random.randint(0, h)

        if forme == 'cercle':
            rayon = random.randint(int(min(h,w)/100), int(min(h,w)/50))
            cv2.circle(fond, (x1, y1), rayon, couleur, -1)

        elif forme == 'rectangle':
            cv2.rectangle(fond, (x1, y1), (x2, y2), couleur, -1)

        elif forme == 'ligne':
            cv2.line(fond, (x1, y1), (x2, y2), couleur, epaisseur)

        elif forme == 'triangle':
            pts = np.array([
                [random.randint(0, w), random.randint(0, h)],
                [random.randint(0, w), random.randint(0, h)],
                [random.randint(0, w), random.randint(0, h)]
            ])
            cv2.fillConvexPoly(fond, pts, couleur)

    return fond


import math
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
        motif_type = np.random.choice(['croix', 'étoile', 'triangle', 'point', 'fleur'])

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

    return masque_result


def cadrage(image, forme='rond', couleur_fond='blanc', formes_fond=False, 
            enluminures=True, motif_type='random', contour_epaisseur = 33, bordures=200):
    """
    Applique un cadrage rond ou losange à une image.

    Args:
        image (np.array): Image d'entrée.
        forme (str): 'rond' ou 'losange'.
        couleur_fond (str): 'blanc' pour un fond blanc, 'noir' pour un fond noir, 'photo' pour mettre une autre photo en fond.
        formes_fond (bool): Si True, ajoute des formes aléatoires sur le fond.
        enluminures (bool): Ajoute des motifs décoratifs autour si True.
        motif_type (str): Type de motif ('croix', 'étoile', 'triangle', 'point', 'fleur', 'random').
        contour_epaisseur (int): Épaisseur du contour blanc rajouté autour de l'image (et éventuellement des enluminures).
        bordures (int): Taille de l'espace (en pixels) entre le bord de l'image et la forme.
    """
    
    if couleur_fond=="photo":
        formes_fond = False

    if enluminures:
        formes_fond = False
        if couleur_fond == 'photo':
            couleur_fond = np.random.choice(['blanc', 'noir'])

    h, w = image.shape[:2]

    masque = np.zeros_like(image, dtype=np.uint8)
    masque.fill(255)

    nb_enluminures = np.random.randint(10, 40)

    if forme == 'rond':
        centre = (w // 2, h // 2)
        rayon_h, rayon_w = int(h/2), int(w/2)
        cv2.ellipse(image, centre, (rayon_h-bordures, rayon_w-bordures), 90, 0, 360, (255, 255, 255), contour_epaisseur*2)
        cv2.ellipse(masque, centre, (rayon_h-bordures, rayon_w-bordures), 90, 0, 360, (0,0,0), -1)

        angle_enlu = 360//nb_enluminures
        contour_enlu = cv2.ellipse2Poly(centre, (rayon_h, rayon_w), 90, 0, 360, angle_enlu)

    elif forme == 'losange':
        points = np.array([
            [w // 2, bordures],          # haut
            [w-bordures, h // 2],          # droite
            [w // 2, h-bordures],          # bas
            [bordures, h // 2]           # gauche
        ])

        points2 = np.array([
            [w // 2, 0],          # haut
            [w, h // 2],          # droite
            [w // 2, h],          # bas
            [0, h // 2]           # gauche
        ])

        contour_enlu = []

        # On récupère les points de la forme losange a intervalles réguliers
        for i in range(-1, len(points2)-1):
            p1 = points2[i]
            p2 = points2[i+1]
            num_points = nb_enluminures//4
            for j in range(num_points):
                t = j / num_points
                point = (1 - t) * p1 + t * p2
                contour_enlu.append(point)

        
        cv2.polylines(image, [points], isClosed=True, color=(255, 255, 255), thickness=contour_epaisseur*2) # Contour blanc
        cv2.fillConvexPoly(masque, points, (0,0,0))

    else:
        raise ValueError("Forme non reconnue : choisir 'rond' ou 'losange'.")
    


    if couleur_fond == 'blanc':
        if formes_fond:
            fond_formes = add_formes_fond(masque.copy(), 'noir', nombre_formes=30)
            image = np.where(masque == 0, image, fond_formes)
        
        elif enluminures:
            fond_enluminures = add_enluminures(masque.copy(), contour_enlu, 'noir', motif_type=motif_type, thickness=contour_epaisseur)
            image = np.where(masque == 0, image, fond_enluminures)

        else:
            image = np.where(masque == 0, image, masque)

    elif couleur_fond == 'noir':
        masque = cv2.bitwise_not(masque)
        if formes_fond:
            fond_formes = add_formes_fond(masque.copy(), 'blanc', nombre_formes=30)
            image = np.where(masque == 255, image, fond_formes)

        elif enluminures:
            fond_enluminures = add_enluminures(masque.copy(), contour_enlu, 'blanc', motif_type=motif_type, thickness=contour_epaisseur)
            image = np.where(masque == 255, image, fond_enluminures)

        else:
            image = np.where(masque == 255, image, masque)

    elif couleur_fond == "photo":
        photo_files = glob.glob(absolutePath+"degradations/datasets/backgrounds/*")
        photo_path = np.random.choice(photo_files)    
        photo_path = photo_path.replace("\\", "/")
        autre_photo = cv2.imread(photo_path)

        if len(autre_photo.shape) != len(image.shape):
            if len(image.shape) == 3:
                autre_photo = cv2.colorChange(autre_photo, cv2.COLOR_GRAY2BGR)
            else:
                autre_photo = cv2.colorChange(autre_photo, cv2.COLOR_BGR2GRAY)

        autre_photo = cv2.resize(autre_photo,(w,h))

        image = np.where(masque == 0, image, autre_photo)
    else:
        raise ValueError("Couleur_fond non reconnue : choisir 'noir', 'blanc' ou 'photo'.")
    return image




import copy
class transforms_non_rectangular_frame(nn.Module):
    def __init__(self, forme=None, couleur_fond=None, formes_fond=None,
                 enluminures=True, motif_type='random', contour_epaisseur = 1, bordures=None):
        super(transforms_non_rectangular_frame, self).__init__()
        self.forme = forme
        self.couleur_fond = couleur_fond
        self.formes_fond = formes_fond
        self.enluminures = enluminures
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
            image_array = np.transpose(image.numpy(), (1, 2, 0)) * 255
            image_array = image_array.astype(np.uint8).copy()

            if self.forme is None:
                forme = random.choice(['rond', 'losange'])
            else:
                forme = self.forme
            
            if self.couleur_fond is None:
                couleur_fond = random.choice(['blanc', 'noir', 'photo'])
            else:
                couleur_fond = self.couleur_fond
            
            if self.formes_fond is None:
                formes_fond = random.choice([True, False])
            else:
                formes_fond = self.formes_fond
            
            if self.enluminures is None:
                enluminures = random.choice([True, False])
            else:
                enluminures = self.enluminures

            if self.bordures is None:
                bordures = random.randint(0, 10)
            else:
                bordures = self.bordures

            
            shape = image_array.shape
            image_array, black_borders = remove_black_borders(image_array)
            image = cadrage(image_array, forme, couleur_fond, formes_fond, enluminures, self.motif_type, self.contour_epaisseur, bordures)
            image = restore_black_borders(shape, image, black_borders)

            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image)
            results[i] = image/255
        
        if one_image:
            results = results.squeeze(0)
        return results