import cv2
from augraphy import *

def stains(img):
    stains = Stains(stains_type="severe_stains",
                stains_blend_method="multiply",stains_blend_alpha=0.1
                )
    img_stains = stains(img)
    
    return img_stains


import torch.nn as nn
class transforms_stains(nn.Module):
    def __init__(self):
        super(transforms_stains, self).__init__()

    def __call__(self, batch):
        for image in batch:
            image = stains(image)
        return batch




if __name__ == "__main__":
        
    # Exemple d'utilisation
    path = "degradations/datasets/original/"
    img = cv2.imread(path+"FRAN_0568_000183_L.jpg")
    stained_img = stains(img)

    cv2.imshow("Stains", stained_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
