import cv2
from augraphy import *
import numpy as np

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
    if img2 == None:
        img2 = img.copy()
    
    if len(img.shape)==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if len(img2.shape)==3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # resize
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

    return blended


import torch.nn as nn
class transforms_bleedthrough(nn.Module):
    def __init__(self, alpha=0.3):
        super(transforms_bleedthrough, self).__init__()
        self.alpha = alpha

    def __call__(self, batch):
        for image in batch:
            image = bleedthrough(image, alpha=self.alpha)
        return batch


if __name__ == "__main__":
        
    # Exemple d'utilisation
    path = "degradations/datasets/original/"
    img = cv2.imread(path+"FRAN_0568_000183_L.jpg")
    img2 = cv2.imread(path+"/FRAN_0568_000220_L.jpg")

    bleedthrough_img = bleedthrough(img,img2,0.2)
    #bleedthrough_img = bleedthrough_augraphy(img)

    # resize image for better visualization
    scale_percent = 10 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    bleedthrough_img = cv2.resize(bleedthrough_img, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow("Bleedthrough", bleedthrough_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
