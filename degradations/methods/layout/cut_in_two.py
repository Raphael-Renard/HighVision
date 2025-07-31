import cv2
import numpy as np
import torch
import torch.nn as nn

def cut_in_two(image, decalage=None, ligne_epaisseur=None):
    hauteur, largeur, _ = image.shape

    if decalage is None:
        if hauteur // 100 > 5:
            decalage = np.random.randint(5, hauteur // 100)
        else:
            decalage = np.random.randint(2, 10)
    if ligne_epaisseur is None:
        ligne_epaisseur = np.random.randint(largeur//60, largeur//40)

    moitié = np.random.randint(largeur // 6, 5*largeur // 6)

    gauche = image[:, :moitié]
    droite = image[:, moitié:]

    # fond blanc
    hauteur_max = hauteur + decalage
    image_coupee = np.ones((hauteur_max, largeur, 3), dtype=np.uint8) * 255

    # décalage vers le haut ou le bas
    decalage_droite = np.random.choice([True, False])
    if decalage_droite:
        image_coupee[0:hauteur, 0:moitié] = gauche
        image_coupee[decalage:decalage + hauteur, moitié:] = droite
    else:
        image_coupee[decalage:decalage + hauteur, 0:moitié] = gauche
        image_coupee[0:hauteur, moitié:] = droite

    # ligne noire dégradée entre les deux moitiés
    intensite_ombre = 0.8
    pliure = np.ones((hauteur_max, largeur), dtype=np.float32)

    for x in range(-ligne_epaisseur//2, ligne_epaisseur//2):
        alpha = 1 - (abs(x) / (ligne_epaisseur/2))
        valeur_ombre = (1 - intensite_ombre * alpha)
        pliure[:, moitié + x] = valeur_ombre

    pliure_rgb = cv2.merge([pliure]*3)

    image_coupee = (image_coupee.astype(np.float32) * pliure_rgb).astype(np.uint8)

    image_coupee = cv2.resize(image_coupee, (largeur, hauteur))
    return image_coupee



class transforms_cut_in_two(nn.Module):
    def __init__(self):
        super(transforms_cut_in_two, self).__init__()

    def __call__(self, batch):
        one_image = False
        if len(batch.shape) == 3:
            batch = batch.unsqueeze(0)
            one_image = True
            
        results = torch.empty_like(batch)
        for i, image in enumerate(batch):
            image_array = np.transpose(np.array(image), (1, 2, 0)) * 255
            image_array = image_array.astype(np.uint8)

            image = cut_in_two(image_array)

            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image)
            results[i] = image/255
        
        if one_image:
            results = results.squeeze(0)
        return results
    
