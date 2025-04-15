import cv2
import torch.nn as nn
import torch
import numpy as np
import random


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


def cadrage(image, forme='cercle', couleur_fond='blanc', formes_fond=True):
    """
    Applique un cadrage rond ou losange à une image.

    Args:
        image (np.array): Image d'entrée.
        forme (str): 'cercle' ou 'losange'.
        couleur_fond (str): 'blanc' pour un fond blanc, 'noir' pour un fond noir.
        formes_fond (bool): Si True, ajoute des formes aléatoires sur le fond.
    """
    

    h, w = image.shape[:2]

    masque = np.zeros_like(image, dtype=np.uint8)
    masque.fill(255)

    if forme == 'cercle':
        centre = (w // 2, h // 2)
        rayon = min(w, h) // 2
        cv2.circle(masque, centre, rayon, (0,0,0), -1)

    elif forme == 'losange':
        points = np.array([
            [w // 2, 0],          # haut
            [w, h // 2],          # droite
            [w // 2, h],          # bas
            [0, h // 2]           # gauche
        ])
        cv2.fillConvexPoly(masque, points, (0,0,0))

    else:
        raise ValueError("Forme non reconnue : choisir 'cercle' ou 'losange'.")
    


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
                forme = random.choice(['cercle', 'losange'])
            else:
                forme = self.forme
            
            if self.couleur_fond is None:
                couleur_fond = random.choice(['blanc', 'noir'])
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
    

