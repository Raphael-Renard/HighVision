import cv2
from augraphy import*

def dirty_rollers(img, line_width_range=(12, 25)):
    dirty_roller = DirtyRollers(line_width_range=line_width_range,
                                scanline_type=0,
                                )

    img_dirty_rollers = dirty_roller(img)
    return img_dirty_rollers


import torch.nn as nn
class transforms_dirty_rollers(nn.Module):
    def __init__(self, line_width_range = (12, 25)):
        super(transforms_dirty_rollers, self).__init__()
        self.line_width_range = line_width_range

    def __call__(self, batch):
        for image in batch:
            image = dirty_rollers(image, self.line_width_range)
        return batch



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
