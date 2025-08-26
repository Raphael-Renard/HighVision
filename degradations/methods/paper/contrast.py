import cv2
import torch.nn as nn
import numpy as np
import torch
from degradations.methods.utils import remove_black_borders, restore_black_borders




def contrast(img):
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
            image = contrast(image_array)
            image = restore_black_borders(shape, image, black_borders)
            image = np.array(image).swapaxes(0,2)
            image = torch.tensor(image)
            results[i] = image / 255
        
        if one_image:
            results = results.squeeze(0)
        return results
    

