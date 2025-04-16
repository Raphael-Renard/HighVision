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


def cadrage(image, forme='rond', couleur_fond='blanc', formes_fond=True, contour_epaisseur = 33):
    """
    Applique un cadrage rond ou losange à une image.

    Args:
        image (np.array): Image d'entrée.
        forme (str): 'rond' ou 'losange'.
        couleur_fond (str): 'blanc' pour un fond blanc, 'noir' pour un fond noir, 'photo' pour mettre une autre photo en fond.
        formes_fond (bool): Si True, ajoute des formes aléatoires sur le fond.
    """
    
    if couleur_fond=="photo":
        formes_fond = False

    h, w = image.shape[:2]

    masque = np.zeros_like(image, dtype=np.uint8)
    masque.fill(255)

    if forme == 'rond':
        centre = (w // 2, h // 2)
        rayon_h, rayon_w = int(h/2), int(w/2)
        cv2.ellipse(image, centre, (rayon_h, rayon_w), 90, 0, 360, (255, 255, 255), contour_epaisseur*2)
        cv2.ellipse(masque, centre, (rayon_h, rayon_w), 90, 0, 360, (0,0,0), -1)

    elif forme == 'losange':
        points = np.array([
            [w // 2, 0],          # haut
            [w, h // 2],          # droite
            [w // 2, h],          # bas
            [0, h // 2]           # gauche
        ])
        
        cv2.polylines(image, [points], isClosed=True, color=(255, 255, 255), thickness=contour_epaisseur*2) # Contour blanc
        cv2.fillConvexPoly(masque, points, (0,0,0))

    else:
        raise ValueError("Forme non reconnue : choisir 'rond' ou 'losange'.")
    


    if couleur_fond == 'blanc':
        if formes_fond:
            fond_formes = add_formes_fond(masque.copy(), 'noir', nombre_formes=30)
            image = np.where(masque == 0, image, fond_formes)
        else:
            image = np.where(masque == 0, image, masque)

    elif couleur_fond == 'noir':
        masque = cv2.bitwise_not(masque)
        if formes_fond:
            fond_formes = add_formes_fond(masque.copy(), 'blanc', nombre_formes=30)
            image = np.where(masque == 255, image, fond_formes)
        else:
            image = np.where(masque == 255, image, masque)

    elif couleur_fond == "photo":
        photo_files = glob.glob(absolutePath+"degradations/datasets/backgrounds/*")
        photo_path = np.random.choice(photo_files)    
        photo_path = photo_path.replace("\\", "/")
        autre_photo = cv2.imdecode(np.fromfile(photo_path, np.uint8), cv2.IMREAD_UNCHANGED)

        if len(autre_photo.shape) != len(image.shape):
            if len(image.shape) == 3:
                autre_photo = cv2.colorChange(autre_photo, cv2.COLOR_BGR2GRAY)
            else:
                autre_photo = cv2.colorChange(autre_photo, cv2.COLOR_GRAY2BGR)

        autre_photo = cv2.resize(autre_photo,(w,h))

        image = np.where(masque == 0, image, autre_photo)
    else:
        raise ValueError("Couleur_fond non reconnue : choisir 'noir', 'blanc' ou 'photo'.")
    return image



class transforms_cadre(nn.Module):
    def __init__(self, forme=None, couleur_fond=None, formes_fond=None):
        super(transforms_cadre, self).__init__()
        self.forme = forme
        self.couleur_fond = couleur_fond
        self.formes_fond = formes_fond

    def __call__(self, batch):
        one_image = False
        if len(batch.shape) == 3: # si on ne passe qu'une image au lieu d'un batch
            batch = batch.unsqueeze(0)
            one_image = True
            
        results = torch.empty_like(batch)
        for i, image in enumerate(batch):
            image_array = np.transpose(np.array(image), (1, 2, 0)).copy() * 255
            image_array = image_array.astype(np.uint8)

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

            image = cadrage(image_array, forme, couleur_fond, formes_fond)
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image)
            results[i] = image/255
        
        if one_image:
            results = results.squeeze(0)
        return results
    

