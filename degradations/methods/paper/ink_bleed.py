from augraphy import *

def ink_bleed(img, intensity_range=(0.4, 0.7)):
    """
    Adds subtle ink bleed to an image to simulate folded paper using augraphy library by python. This is a wrapper function for modularity purposes.

    """

    inkbleed = InkBleed(intensity_range,
                    kernel_size=(5, 5),
                    severity=(0.2, 0.4)
                        )

    img_inkbleed = inkbleed(img)

    return img_inkbleed


import torch.nn as nn
class transforms_ink_bleed(nn.Module):
    def __init__(self, intensity_range=(0.4, 0.7)):
        super(transforms_ink_bleed, self).__init__()
        self.intensity_range = intensity_range

    def __call__(self, batch):
        for image in batch:
            image = ink_bleed(image, self.intensity_range)
        return batch

