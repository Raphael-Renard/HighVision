from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import torch
import copy
import torch.nn as nn
import cv2

def faded(img):
    img = Image.fromarray(img)
    img = ImageEnhance.Color(img).enhance(0.2)
    img = ImageEnhance.Contrast(img).enhance(0.5)
    img = ImageEnhance.Brightness(img).enhance(1.1)

    img = img.filter(ImageFilter.GaussianBlur(radius=1))

    # ajouter un voile blanc semi-transparent pour "effacer"
    voile = Image.new("RGB", img.size, (255, 255, 255))
    img = Image.blend(img, voile, alpha=0.1)

    img = np.array(img)
    return img



class transforms_faded(nn.Module):
    def __init__(self):
        super(transforms_faded, self).__init__()

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

            image = faded(image_array)

            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image)
            results[i] = image/255
        
        if one_image:
            results = results.squeeze(0)
        return results
    

