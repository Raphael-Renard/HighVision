import cv2
from augraphy import*
import torch.nn as nn
import torch
import numpy as np

def dirty_rollers(img, line_width_range=(25, 35)):
    dirty_roller = DirtyRollers(line_width_range=line_width_range,
                                scanline_type=0,
                                )

    img_dirty_rollers = dirty_roller(img)
    return img_dirty_rollers



class transforms_dirty_rollers(nn.Module):
    def __init__(self, line_width_range = (8, 10)):
        super(transforms_dirty_rollers, self).__init__()
        self.line_width_range = line_width_range

    def __call__(self, batch):
        results = torch.empty_like(batch)
        for i, image in enumerate(batch):
            image_array = np.array(image).swapaxes(0,2) * 255
            mask = np.where(image_array==0)
            image = dirty_rollers(image_array, self.line_width_range)
            image[mask] = 0
            image = np.array(image).swapaxes(0,2)
            image = torch.tensor(image)
            results[i] = image / 255
        return results


"""
if __name__ == "__main__":
    path = "degradations/datasets/original/"
    img = cv2.imread(path+"FRAN_0568_000020_L.jpg")
    img = dirty_rollers(img)

    
    # resize image for better visualization
    scale_percent = 10 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow("dirty rollers", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""

    
if __name__ =="__main__":
    image_path = "C:/Users/rapha/Documents/Cours/Master/Stage/HighVision/degradations/results/2K2476_16_01.jpg"
    #image_path = "C:/Users/rapha/Documents/Cours/Master/Stage/Data/Sena/FRAN_0568_11AR_699/FRAN_0568_000014_L.jpg"
    img = cv2.imread(image_path)
    img = dirty_rollers(img,(8,10))
    cv2.imwrite("dirty_rollers.jpg",img)