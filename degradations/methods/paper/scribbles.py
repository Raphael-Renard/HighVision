from augraphy import*
import cv2

def scribbles(img):
    h,w,_ = img.shape
    thickness = max(h,w)//500
    scribbles = Scribbles(scribbles_thickness_range=(thickness, thickness+3), scribbles_color="random")

    img_scribbles = scribbles(img)
    return img_scribbles


import torch.nn as nn
class transforms_scribbles(nn.Module):
    def __init__(self):
        super(transforms_scribbles, self).__init__()

    def __call__(self, batch):
        for image in batch:
            image = scribbles(image)
        return batch



if __name__ == "__main__":
    
    # Exemple d'utilisation
    path = "degradations/datasets/original/"
    img = cv2.imread(path+"FRAN_0568_000183_L.jpg")

    erased_img = scribbles(img)

    cv2.imshow("Erased Effect", erased_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
