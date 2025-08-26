import cv2
from augraphy import *
import torch.nn as nn
import numpy as np
import torch


def stains(img):
    stain = Stains(stains_type="rough_stains",
                stains_blend_method="multiply",stains_blend_alpha=0.8
                )
    img_stains = stain(img)
    
    return img_stains


class transforms_stains(nn.Module):
    def __init__(self):
        super(transforms_stains, self).__init__()

    def __call__(self, batch):
        one_image = False
        if len(batch.shape) == 3: # si on ne passe qu'une image au lieu d'un batch
            batch = batch.unsqueeze(0)
            one_image = True

        results = torch.empty_like(batch)
        for i, image in enumerate(batch):
            image_array = np.array(image).swapaxes(0,2) * 255
            mask = np.where(image_array==0)
            image = stains(image_array)
            image[mask]=0
            image = np.array(image).swapaxes(0,2)
            image = torch.tensor(image)
            results[i] = image / 255
        
        if one_image:
            results = results.squeeze(0)
        return results


