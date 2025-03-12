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
        results = torch.empty_like(batch)
        for i, image in enumerate(batch):
            image_array = np.array(image).swapaxes(0,2) * 255
            mask = np.where(image_array==0)
            image = stains(image_array)
            image[mask]=0
            image = np.array(image).swapaxes(0,2)
            image = torch.tensor(image)
            results[i] = image / 255
        return results



"""
if __name__ == "__main__":
    # Exemple d'utilisation
    path = "corpus/lipade_groundtruth/unique/"
    img = cv2.imread(path+"2K2476_02_01.jpg")

     # resize image for better visualization
    scale_percent = 20 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    stained_img = stains(img)

   

    cv2.imshow("Stains", stained_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""


if __name__ =="__main__":
    image_path = "C:/Users/rapha/Documents/Cours/Master/Stage/HighVision/degradations/results/2K2476_16_01.jpg"
    #image_path = "C:/Users/rapha/Documents/Cours/Master/Stage/Data/Sena/FRAN_0568_11AR_699/FRAN_0568_000014_L.jpg"
    img = cv2.imread(image_path)
    img = stains(img)
    cv2.imwrite("stains.jpg",img)