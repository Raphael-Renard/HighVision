from augraphy import*
import cv2
import torch.nn as nn
import torch
import numpy as np

def scribbles(img, thickness=1, size_range=(50,100)):
    h,w,_ = img.shape
    if thickness is None:
        thickness = max(h,w)//500
        scribbles = Scribbles(scribbles_thickness_range=(thickness, thickness+3),
                              scribbles_count_range=(1, 4))
    else:
        scribbles = Scribbles(scribbles_thickness_range=(thickness, thickness+1), scribbles_size_range=size_range,
                              scribbles_count_range=(1, 4))
    img_scribbles = scribbles(img.astype(np.uint8))
    return img_scribbles



class transforms_scribbles(nn.Module):
    def __init__(self, thickness=1, size_range=(20,50)):
        super(transforms_scribbles, self).__init__()
        self.thickness = thickness
        self.size_range = size_range

    def __call__(self, batch):
        results = torch.empty_like(batch)
        for i, image in enumerate(batch):
            image_array = np.array(image).swapaxes(0,2) * 255
            mask = np.where(image_array==0)
            image = scribbles(image_array, self.thickness, self.size_range)
            image[mask]=0
            image = np.array(image).swapaxes(0,2)
            image = torch.tensor(image)
            results[i] = image / 255
        return results



if __name__ == "__main__":
    
    # Exemple d'utilisation
    path = "degradations/datasets/original/"
    img = cv2.imread(path+"FRAN_0568_000183_L.jpg")

    erased_img = scribbles(img)

    cv2.imshow("Erased Effect", erased_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
