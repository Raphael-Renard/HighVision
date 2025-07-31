import cv2
import numpy as np
import torch
import torch.nn as nn

def book(image, type_distorsion="asymetrique", intensite_ombre=0.8, largeur_pliure=None, amplitude=None):
    """
    Donne un effet livre à l'image en ajoutant une ombre au milieu et une distorsion, comme 2 pages courbées.

    :param image: Image d'entrée (numpy array).
    :param intensite_ombre: Intensité de l'ombre (0 à 1).   
    :param largeur_pliure: Largeur de la pliure (en pixels) : taille de l'ombre noire au milieu de l'image.
    :param amplitude: Amplitude de la distorsion (en pixels) : à quel point les pages sont courbées.
    """
    
    hauteur, largeur, _ = image.shape
    centre_x = largeur // 2

    amplitude = np.random.randint(hauteur//51, hauteur//18) if amplitude is None else amplitude
    largeur_pliure = np.random.randint(amplitude*2, amplitude*10) if largeur_pliure is None else largeur_pliure

    # Créer une ombre en dégradé pour simuler la pliure
    pliure = np.ones((hauteur, largeur), dtype=np.float32)

    for x in range(-largeur_pliure//2, largeur_pliure//2):
        alpha = 1 - (abs(x) / (largeur_pliure/2))  # Dégradé centré
        valeur_ombre = (1 - intensite_ombre * alpha)
        pliure[:, centre_x + x] = valeur_ombre

    # Étendre le pli pour qu'il ait 3 canaux (RGB)
    pliure_rgb = cv2.merge([pliure]*3)

    # Appliquer l'effet de pliure
    image_pliee = (image.astype(np.float32) * pliure_rgb).astype(np.uint8)

    # Ajouter une distorsion horizontale (courbure des pages)
    def distorsion_horizontale(x, frequence=1):
        if type_distorsion == "sin":
            if x < centre_x:
                x = largeur//2 + x
        else:
            if x > centre_x:
                x = largeur - x
        norm_x = (x - centre_x/4) / (largeur//4)

        if type_distorsion == "sin":
            return int(amplitude * np.sin(2 * np.pi * x / largeur * frequence))
       
        elif type_distorsion == "asymetrique":
            small_amp = 3*amplitude // 4
            return int(small_amp * (norm_x ** 3) - small_amp * norm_x)
        
        elif type_distorsion == "plateau":
            if norm_x > end_plateau:
                return int(amplitude * (norm_x - end_plateau))
            elif start_plateau <= norm_x <= end_plateau:
                return 0
            else:
                return int(amplitude * (start_plateau - norm_x))
        else:
            raise ValueError("Type de distorsion non reconnu. Choisissez entre 'asymetrique', 'sin' ou 'plateau'.")


    image_finale = np.zeros_like(image_pliee)
    if type_distorsion == "plateau":
        start_plateau = np.random.uniform(0.6,0.9)
        end_plateau = np.random.uniform(1,1.3)
    for x in range(largeur):
        decalage = distorsion_horizontale(x)
        ligne_decalee = np.roll(image_pliee[:, x], decalage, axis=0)
        image_finale[:, x] = ligne_decalee

    return image_finale





class transforms_book(nn.Module):
    def __init__(self, intensite_ombre=0.8):
        super(transforms_book, self).__init__()
        self.intensite_ombre = intensite_ombre

    def __call__(self, batch):
        one_image = False
        if len(batch.shape) == 3: # si on ne passe qu'une image au lieu d'un batch
            batch = batch.unsqueeze(0)
            one_image = True

        results = torch.empty_like(batch)
        for i, image in enumerate(batch):
            image_array = np.array(image).swapaxes(0,2) * 255
            type_distorsion = np.random.choice(["asymetrique", "sin", "plateau"])
            image = book(image_array, type_distorsion, self.intensite_ombre)
            image = np.array(image).swapaxes(0,2)
            image = torch.tensor(image)
            results[i] = image / 255
        
        if one_image:
            results = results.squeeze(0)
        return results

