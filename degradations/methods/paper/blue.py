import numpy as np
import torch
import torch.nn as nn
import cv2


def blue(img):
    if len(img.shape)==3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    tinted = np.stack([gray, gray, gray], axis=-1).astype(np.int16)

    # Définir un masque pour les pixels sombres
    mask = gray < 128

    # Appliquer une teinte bleue sur ces pixels sombres
    tinted[mask, 0] = np.clip(tinted[mask, 0] + 60, 0, 255)   # Blue 
    tinted[mask, 1] = np.clip(tinted[mask, 1] + 10, 0, 255)   # Green 
    tinted[mask, 2] = np.clip(tinted[mask, 2] , 0, 255)   # Red 

    return tinted.astype(np.uint8)



class transforms_blue(nn.Module):
    def __init__(self):
        super(transforms_blue, self).__init__()

    def __call__(self, batch):
        one_image = False
        if len(batch.shape) == 3: # si on ne passe qu'une image au lieu d'un batch
            batch = batch.unsqueeze(0)
            one_image = True
            
        results = torch.empty_like(batch)
        for i, image in enumerate(batch):
            image_array = np.transpose(np.array(image), (1, 2, 0)) * 255
            image_array = image_array.astype(np.uint8)

            image = blue(image_array)

            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image)
            results[i] = image/255
        
        if one_image:
            results = results.squeeze(0)
        return results
    

