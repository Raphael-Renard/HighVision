import sys
path = "../../.."
if path not in sys.path:
    sys.path.insert(0, path)

from data_retrieval import lipade_groundtruth
from PIL import Image
from transformers import FlavaImageProcessor, FlavaImageModel
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from data_retrieval.tools.data_loader import getDataLoader
from torch.utils.data import DataLoader
import torch.optim as optim
from degradations.methods import transforms_atkinson_dithering, transforms_bayer_halftoning, transforms_floyd_steinberg_halftoning, transforms_drawing, transforms_erased_element, transforms_paint, transforms_non_rectangular_frame, transforms_patchwork, transforms_photo_montage, transforms_picture_overlay, transforms_text_overlay, transforms_dirty_rollers, transforms_add_gaussian_noise, transforms_add_salt_and_pepper_noise, transforms_bleedthrough, transforms_contrast, transforms_crumpled_paper, transforms_folded_paper, transforms_ink_bleed, transforms_book, transforms_stains, transforms_scribbles, transforms_torn_paper


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


corpus = "lipade_groundtruth"
rawPath = "../results/raw/" + corpus + "/"


x,_,y = lipade_groundtruth.getDataset(mode = 'unique', uniform=True)

for i in range(len(x)):
    try:
        x[i] = Image.open(x[i]).convert('RGB')
    except:
        print("Error loading image:", x[i])

images = np.array(x)


class transforms_SepiaFilter(nn.Module):
    def __init__(self):
        super(transforms_SepiaFilter, self).__init__()

    def __call__(self, batch):
        sepia_filter = torch.tensor([[0.393, 0.769, 0.189],
                                     [0.349, 0.686, 0.168],
                                     [0.272, 0.534, 0.131]], device=batch.device)
        batch = torch.einsum('ijkl,mj->imkl', batch, sepia_filter)
        return batch.clamp(0, 1)

transform_1by1 = transforms.Compose([
        transforms.RandomChoice([
            transforms.RandomResizedCrop(size=images.shape[2], scale=(1/2, 1), ratio=(1, 1)),
            transforms_floyd_steinberg_halftoning(),
            transforms_atkinson_dithering(),
            transforms_bayer_halftoning(),
            transforms_picture_overlay(),
            transforms_text_overlay(),
            transforms_torn_paper(),
            transforms_erased_element(),
            transforms_add_gaussian_noise(),
            transforms_add_salt_and_pepper_noise(),
            transforms_dirty_rollers(),
            transforms_stains(),
            transforms_ink_bleed(),
            transforms_bleedthrough(),
            transforms_crumpled_paper(),
            transforms_folded_paper(),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms_SepiaFilter()]),
        transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5),
])


for i, image in tqdm(enumerate(images)):
    image = transforms.ToTensor()(image)
    for j in range(10):
        degraded_image = transform_1by1(image).numpy()
        degraded_image.save(f"{i}/{j}.jpg")

