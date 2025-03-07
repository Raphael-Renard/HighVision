import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import sys
import os

image_path = "C:/Users/rapha/Documents/Cours/Master/Stage/HighVision/degradations/results/2K2476_16_01.jpg"
image = Image.open(image_path).convert("RGB")
to_tensor = transforms.ToTensor()
original_image = to_tensor(image)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from methods.halftoning.floyd_steinberg import transforms_floyd_steinberg_halftoning
from methods.halftoning.atkinson import transforms_atkinson_dithering
from methods.noise.gaussian_noise import transforms_add_gaussian_noise
from methods.noise.salt_and_pepper import transforms_add_salt_and_pepper_noise
from methods.noise.dirty_rollers import transforms_dirty_rollers
from methods.paper.ink_bleed import transforms_ink_bleed  
from methods.paper.crumpled_paper import transforms_crumpled_paper
from methods.paper.folded_paper import transforms_folded_paper
from methods.paper.bleedthrough import transforms_bleedthrough
from methods.paper.scribbles import transforms_scribbles
from methods.paper.stains import transforms_stains 
from methods.human_corrections.erased_element import transforms_erased_element
from methods.layout.picture_overlay import transforms_picture_overlay
from methods.layout.text_overlay import transforms_text_overlay 


class transforms_SepiaFilter(nn.Module):
    def __init__(self):
        super(transforms_SepiaFilter, self).__init__()

    def __call__(self, batch):
        sepia_filter = torch.tensor([[0.393, 0.769, 0.189],
                                     [0.349, 0.686, 0.168],
                                     [0.272, 0.534, 0.131]], device=batch.device)
        batch = torch.einsum('ijkl,mj->imkl', batch, sepia_filter)
        return batch.clamp(0, 1)

class transforms_Rotate(nn.Module):
    def __init__(self):
        super(transforms_Rotate, self).__init__()

    def __call__(self, batch):
        batch = batch.movedim(2,3)    
        return batch
    

# Liste des transformations
transformations = [
    ("Floyd-Steinberg Halftoning", transforms_floyd_steinberg_halftoning(128)),
    ("Atkinson Dithering", transforms_atkinson_dithering(128)),
    ("Gaussian Noise", transforms_add_gaussian_noise()),
    ("Salt & Pepper Noise", transforms_add_salt_and_pepper_noise()),
    ("Dirty Rollers", transforms_dirty_rollers((8,10))),
    ("Ink Bleed", transforms_ink_bleed()),
    ("Crumpled Paper", transforms_crumpled_paper()),
    ("Folded Paper", transforms_folded_paper(0.4)),
    ("Bleedthrough", transforms_bleedthrough()),
    ("Scribbles", transforms_scribbles()),
    ("Stains", transforms_stains()),
    ("Erased Element", transforms_erased_element()),
    ("Picture Overlay", transforms_picture_overlay()),
    ("Text Overlay", transforms_text_overlay()),
    ("Grayscale", transforms.Grayscale(num_output_channels=3)),
    ("Sepia Filter", transforms_SepiaFilter()),
    ("Rotate", transforms_Rotate()),
    ("Flip H", transforms.RandomHorizontalFlip(1)),
    ("Flip V", transforms.RandomVerticalFlip(1)),
    ("Resized Crop", transforms.RandomResizedCrop(size=original_image.shape[0], scale=(2/5, 1), ratio=(1, 1)))
]

# Appliquer les transformations
fig, axes = plt.subplots(4, 5, figsize=(20, 30))
axes = axes.ravel()

for i, (name, transform) in enumerate(transformations):
    print(name)
    if i >= len(axes):  # Limiter si trop de transformations
        break
    transformed_image = transform(original_image.unsqueeze(0)).squeeze(0)
    if isinstance(transformed_image, torch.Tensor):
        transformed_image = transforms.ToPILImage()(transformed_image)
    
    axes[i].imshow(transformed_image)
    axes[i].set_title(name, fontsize=10)
    axes[i].axis("off")

plt.show()
