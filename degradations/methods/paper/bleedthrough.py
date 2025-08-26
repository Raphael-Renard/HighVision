import cv2
from augraphy import *
import numpy as np
import torch.nn as nn
import torch

def bleedthrough_augraphy(img,alpha=0.3):
    """
    Simulate the effect of the other side of the paper page bleeding through 
    (as if the same image was on the other side)

    Params:
        alpha: intensity of the bleedthrough
    """
    bt = BleedThrough(color_range=(0, 50),
                                ksize=(17, 17),
                                sigmaX=0,
                                alpha=alpha,
                                offsets=(10, 10),
                            )

    img_bleedthrough = bt(img)
    return img_bleedthrough


def bleedthrough(img, img2=None, alpha=0.3):
    """
    Simulate the effect of the other side of the paper page bleeding through with another image

    Params:
        img2: the image on the other side of the paper
        alpha: intensity of the bleedthrough
    """
    if img2 is None:
        img2 = img.copy()
    
    if len(img.shape)==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if len(img2.shape)==3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # resize img2
    if img.shape[0] <= img2.shape[0]:
        img2 = img2[:img.shape[0]]
    else:
        white = np.full((img.shape[0], img2.shape[1]), 255, dtype="uint8")
        white[:img2.shape[0]] = img2
        img2 = white

    if img.shape[1] <= img2.shape[1]:
        img2 = img2[:,:img.shape[1]]
    else:
        white = np.full((img2.shape[0], img.shape[1]), 255, dtype="uint8")
        white[:,:img2.shape[1]] = img2
        img2 = white


    # rotate
    img2 = cv2.flip(img2, 1)

    # ink bleed
    inkbleed = InkBleed(intensity_range=(0.4, 0.7),
                    kernel_size=(5, 5),
                    severity=(0.2, 0.4)
                    )
    img2_inkbleed = inkbleed(img2)

    # we only want to keep the darker colors of img2 (white doesn't bleedthrough)
    index_white = np.where(img2_inkbleed>80)

    # blend
    blended = cv2.addWeighted(img,(1-alpha),img2_inkbleed,alpha,0)
    
    # delete the white bleedthrough
    blended[index_white] = img[index_white]

    blended = cv2.cvtColor(blended,cv2.COLOR_GRAY2BGR)
    return blended



class transforms_bleedthrough(nn.Module):
    def __init__(self, alpha=0.3):
        super(transforms_bleedthrough, self).__init__()
        self.alpha = alpha

    def __call__(self, batch):
        one_image = False
        if len(batch.shape) == 3: # si on ne passe qu'une image au lieu d'un batch
            batch = batch.unsqueeze(0)
            one_image = True
            
        results = torch.empty_like(batch)
        for i, image in enumerate(batch):
            image_array = np.array(image).swapaxes(0,2) * 255
            mask = np.where(image_array==0)
            image = bleedthrough(image_array, alpha=self.alpha)
            image[mask]=0
            image = np.array(image).swapaxes(0,2)
            image = torch.tensor(image) / 255
            results[i] = image
        
        if one_image:
            results = results.squeeze(0)
        return results


