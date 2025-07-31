from augraphy import *
import torch.nn as nn
import torch
import numpy as np
import cv2


def ink_bleed(img, intensity_range=(0.4, 0.7)):
    """
    Adds subtle ink bleed to an image to simulate folded paper using augraphy library by python. This is a wrapper function for modularity purposes.

    """

    inkbleed = InkBleed(intensity_range,
                    kernel_size=(5, 5),
                    severity=(0.5, 0.8)
                        )

    img_inkbleed = inkbleed(img)

    return img_inkbleed



class transforms_ink_bleed(nn.Module):
    def __init__(self, intensity_range=(0.4, 0.7)):
        super(transforms_ink_bleed, self).__init__()
        self.intensity_range = intensity_range

    def __call__(self, batch):
        one_image = False
        if len(batch.shape) == 3: # si on ne passe qu'une image au lieu d'un batch
            batch = batch.unsqueeze(0)
            one_image = True

        results = torch.empty_like(batch)
        for i, image in enumerate(batch):
            image_array = np.array(image).swapaxes(0,2) * 255
            mask = np.where(image_array==0)
            image = ink_bleed(image_array, self.intensity_range)
            image[mask]=0
            image = np.array(image).swapaxes(0,2)
            image = torch.tensor(image)
            results[i] = image / 255
        
        if one_image:
            results = results.squeeze(0)
        return results

