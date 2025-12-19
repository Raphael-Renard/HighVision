import sys
path = "../../.."
if path not in sys.path:
    sys.path.insert(0, path)


import matplotlib.pyplot as plt
import numpy as np
from data_retrieval import lipade_groundtruth
from data_retrieval.tools.data_loader import getDataLoader
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from data_retrieval import lipade_groundtruth
from clustering.clustering import getPredictionFromThreshold
import clustering.evaluators as evaluators
import csv
import os

from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

import clip
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(device)

optimizerFunc = optim.Adam
temperature = 0.07
learningRate = 3e-4
batch_size = 128 # metttre plus grand
workers = 2
extract = 800
epochs = 20
corpus = "lipade_groundtruth"
resultsPath = "./representation/multimodal/results/distance/" + corpus + "/"
weightPath = "./representation/multimodal/results/weights/"
lossPath = "./representation/multimodal/results/loss/"


csv_filename = "results_slip.csv"


torch.cuda.empty_cache()




# Test
xSim,_,ySim = lipade_groundtruth.getDataset(mode = 'similar') #, uniform=True)
_,_,y_test = lipade_groundtruth.getDataset(mode="similar")

imagesSim = []
for i in range(len(xSim)):
    try:
        imagesSim.append(Image.open(xSim[i]).convert('RGB'))
    except:
        print("Error loading image:", xSim[i])



def _convert_image_to_rgb(image):
    return image.convert("RGB")

trans_prepro = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

testLoader = getDataLoader(imagesSim, None, trans_prepro, False, batch_size, shuffle=False, num_workers=2)


# Train/val
# Images
x,m,_ = lipade_groundtruth.getDataset(mode="unique") #, uniform=True)

is_recto = np.array(m[2])
x = np.array(x)[is_recto]

# Captions
captions = m[1]
images_per_captions = {}
images = []
for file,prompt in captions.keys():
    images_per_captions[prompt] = []

for i in range(len(x)):
    f = '/'.join(x[i].split('/')[-2:])
    if (f, prompt) not in captions.keys():
        images.append(x[i])

x = x.tolist()
for im in images:
    x.remove(im)

for prompt in images_per_captions.keys():
    for i in range(len(x)):
        f = '/'.join(x[i].split('/')[-2:])
        images_per_captions[prompt].append(captions[(f, prompt)])

class CustomDataset(Dataset):
    def __init__(self, image_paths, captions):
        self.image_paths = image_paths
        self.captions = captions
        self.transform = trans_prepro
        #transforms.Compose([
        #    transforms.Resize((224, 224)),
        #    transforms.ToTensor()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, image_path, self.captions[idx], idx
    



captions = images_per_captions["a black and white photo of "]

train_images, val_images, train_captions, val_captions = train_test_split(
    x, captions, test_size=0.1, random_state=42
)

train_loader = DataLoader(CustomDataset(train_images, train_captions), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(CustomDataset(val_images, val_captions), batch_size=batch_size, shuffle=False)




from degradations.methods import transforms_atkinson_dithering, transforms_bayer_halftoning, transforms_floyd_steinberg_halftoning, transforms_drawing, transforms_erased_element, transforms_paint, transforms_non_rectangular_frame, transforms_patchwork, transforms_photo_montage, transforms_picture_overlay, transforms_text_overlay, transforms_dirty_rollers, transforms_add_gaussian_noise, transforms_add_salt_and_pepper_noise, transforms_bleedthrough, transforms_contrast, transforms_crumpled_paper, transforms_folded_paper, transforms_ink_bleed, transforms_book, transforms_stains, transforms_scribbles, transforms_torn_paper


class transforms_SepiaFilter(nn.Module):
    def __init__(self):
        super(transforms_SepiaFilter, self).__init__()

    def __call__(self, batch):
        sepia_filter = torch.tensor([[0.393, 0.769, 0.189],
                                     [0.349, 0.686, 0.168],
                                     [0.272, 0.534, 0.131]], device=batch.device)
        batch = torch.einsum('ijkl,mj->imkl', batch, sepia_filter)
        return batch.clamp(0, 1)



list_degrads = [
        # halftone
        transforms_floyd_steinberg_halftoning(),
        transforms_atkinson_dithering(),
        transforms_bayer_halftoning(),
        # layout
        transforms_picture_overlay(),
        transforms_text_overlay(),
        transforms_non_rectangular_frame(),
        transforms_photo_montage(),
        # human
        transforms_erased_element(),
        #transforms_drawing(),
        #transforms_paint(),
        # noise
        transforms_add_gaussian_noise(),
        transforms_add_salt_and_pepper_noise(),
        transforms_dirty_rollers(),
        # stains
        #transforms_scribbles(),
        transforms_stains(),
        transforms_ink_bleed(),
        transforms_bleedthrough(),
        # texture
        transforms_crumpled_paper(),
        transforms_folded_paper(),
        transforms_torn_paper(),
        # sepia
        transforms_SepiaFilter()]



transform = transforms.Compose([
            transforms.RandomChoice([
                transforms.RandomResizedCrop(size=224, scale=(1/2, 1), ratio=(1, 1)),
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomVerticalFlip(p=1),
                *list_degrads]),
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=9)], p=0.5),
        ])



# Charger CLIP
"""
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32") #.to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# Projection SimCLR
class ProjectionMLP(nn.Module):
    def __init__(self, in_dim=512, out_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )
    def forward(self, x):
        return self.mlp(x)

proj = ProjectionMLP(in_dim=512) #.to(device)

# Optimizer
optimizer = optim.AdamW(
    list(clip_model.vision_model.parameters()) + list(proj.parameters()),
    lr=5e-6
)+ list(proj.parameters())
"""

model, _ = clip.load("ViT-B/32", device)
optimizer = optim.AdamW(
    list(model.parameters()),
    lr=5e-6
)

temperature = 0.07
#lambda_clip = 1.0
#lambda_simclr = 1.0






def infoNCEloss(z1, z2, t=1):
    z = torch.cat([z1, z2], dim=0)

    s = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)
    exp_s = torch.exp(s / t)
    indicatorMask = torch.eye(s.shape[0], dtype=torch.bool, device=z.device) # True on diagonal, False elsewhere
    exp_s = exp_s.masked_fill(indicatorMask, 0)

    numerator = F.cosine_similarity(z1, z2, dim=-1)      # for z1
    numerator = torch.cat([numerator, numerator], dim=0) # for z2
    numerator = torch.exp(numerator / t)

    denominator = exp_s.sum(dim=1)

    l = -torch.log(numerator / denominator)
    return l.mean()


# CLIP Loss
def clip_similarity_loss(image_embeds, text_embeds, t):
    image_embeddings = F.normalize(image_embeds, dim=1)
    text_embeddings = F.normalize(text_embeds, dim=1)

    logits_per_image = image_embeddings @ text_embeddings.T
    logits_per_text = text_embeddings @ image_embeddings.T

    logits_per_image /= t
    logits_per_text /= t

    labels = torch.arange(len(image_embeddings)).to(image_embeddings.device)

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)

    return (loss_i + loss_t) / 2


def compute_validation_losses(val_loader, clip_model, proj, transform):
    clip_model.eval()
    proj.eval()

    sim_losses = []
    clip_losses = []

    with torch.no_grad():
        for (img, img_paths, captions, idx) in val_loader:

            images = [Image.open(p).convert("RGB") for p in img_paths]

            # --- SIMCLR ---
            x = img
            x2 = transform(img)

            v1 = clip_model.get_image_features(pixel_values=x)
            v2 = clip_model.get_image_features(pixel_values=x2)

            z1 = proj(v1)
            z2 = proj(v2)

            sim_loss = infoNCEloss(z1, z2, temperature)
            sim_losses.append(sim_loss.item())

            # --- CLIP ---
            inputs = processor(images=images, text=captions,
                               return_tensors="pt", padding=True)
            outputs = clip_model(**inputs)

            img_emb = outputs.image_embeds
            txt_emb = outputs.text_embeds

            clip_loss = clip_similarity_loss(img_emb, txt_emb, temperature)
            clip_losses.append(clip_loss.item())

    clip_model.train()
    proj.train()

    return np.mean(sim_losses), np.mean(clip_losses), np.mean(sim_losses) + np.mean(clip_losses)




def compute_validation_losses_new(val_loader, clip_model, transform):
    clip_model.eval()

    sim_losses = []
    clip_losses = []

    with torch.no_grad():
        for (img, img_paths, captions, idx) in val_loader:

            images = [Image.open(p).convert("RGB") for p in img_paths]

            # --- SIMCLR ---
            x = img
            x2 = transform(img)

            v1 = clip_model.encode_image(x)
            v2 = clip_model.encode_image(x2)


            sim_loss = infoNCEloss(v1, v2, temperature)
            sim_losses.append(sim_loss.item())

            # --- CLIP ---
            text = clip.tokenize(list(captions))#.to(device)
            img_emb, txt_emb = clip_model(img, text)

            clip_loss = clip_similarity_loss(img_emb, txt_emb, temperature)
            clip_losses.append(clip_loss.item())

    clip_model.train()

    return np.mean(sim_losses), np.mean(clip_losses), np.mean(sim_losses) + np.mean(clip_losses)






def train_joint(train_loader, val_loader, transform, name, epochs=10, lambda_clip = 1.0, lambda_simclr = 1.0):
    clip_model.train()
    proj.train()
    losses_simclr_all_epoch = []
    losses_clip_all_epoch = []
    losses_simclr_val_all_epoch = []
    losses_clip_val_all_epoch = []
    losses_sum_val_all_epoch = []

    if not os.path.exists(weightPath+f"SLIP_joint_{name}"):
        os.mkdir(weightPath+f"SLIP_joint_{name}")

    for epoch in range(epochs):
        losses_simclr = []
        losses_clip = []

        for (img, img_paths, captions, idx) in tqdm(train_loader):
            images = [Image.open(p).convert("RGB") for p in img_paths]
            
            # --- SIMCLR PART ---
            
            x = img #.to(device)
            x2 = transform(img) #.to(device)

            v1 = clip_model.get_image_features(pixel_values=x)
            v2 = clip_model.get_image_features(pixel_values=x2)

            z1 = proj(v1)
            z2 = proj(v2)

            loss_simclr = infoNCEloss(z1, z2, temperature)

            # --- CLIP PART ---
            
            inputs = processor(images=images, text=captions, return_tensors="pt", padding=True) #.to(device)

            outputs = clip_model(**inputs)
            img_emb = outputs.image_embeds
            txt_emb = outputs.text_embeds

            loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)

            # --- TOTAL LOSS ---
            loss_total = lambda_simclr * loss_simclr + lambda_clip * loss_clip
            
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            losses_simclr.append(loss_simclr.item())
            losses_clip.append(loss_clip.item())
        
        # --- Validation ---
        sim_val, clip_val, sum_val = compute_validation_losses(val_loader, clip_model, proj, transform)

        print(f"[Epoch {epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
        losses_simclr_all_epoch.append(np.mean(losses_simclr))
        losses_clip_all_epoch.append(np.mean(losses_clip))
        losses_simclr_val_all_epoch.append(sim_val)
        losses_clip_val_all_epoch.append(clip_val)
        losses_sum_val_all_epoch.append(sum_val)

        torch.save({
            "clip": clip_model.state_dict(),
            "proj": proj.state_dict()
        }, weightPath+f"SLIP_joint_{name}/{epoch}.pth")
    

    plt.plot(losses_simclr_all_epoch, label="InfoNCE Loss", color='b')
    plt.plot(losses_clip_all_epoch, label="CLIP Loss", color='g')
    plt.plot(losses_simclr_val_all_epoch, label="Val InfoNCE", color='r')
    plt.plot(losses_clip_val_all_epoch,   label="Val CLIP", color='orange')
    plt.plot(losses_sum_val_all_epoch,   label="Val CLIP+InfoNCE", color='yellow')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("SLIP Training: SimCLR and CLIP")
    plt.legend()
    plt.savefig(lossPath+f"loss_slip_joint_{name}.png")
    plt.close()






def train_joint_new(train_loader, transform, name, epochs=10,lambda_clip = 1.0, lambda_simclr = 1.0):
    model.train()
    losses_simclr_all_epoch = []
    losses_clip_all_epoch = []
    losses_simclr_val_all_epoch = []
    losses_clip_val_all_epoch = []
    losses_sum_val_all_epoch = []


    if not os.path.exists(weightPath+f"SLIP_joint_{name}"):
        os.mkdir(weightPath+f"SLIP_joint_{name}")

    for epoch in range(epochs):
        losses_simclr = []
        losses_clip = []

        for (img, img_paths, captions, idx) in tqdm(train_loader):

            # --- SIMCLR PART ---
            
            x = img #.to(device)
            x2 = transform(img) #.to(device)

            v1 = model.encode_image(x)
            v2 = model.encode_image(x2)

            loss_simclr = infoNCEloss(v1, v2, temperature)

            # --- CLIP PART ---
            
            text = clip.tokenize(list(captions))#.to(device)
            img_emb, txt_emb = model(img, text)

            loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)

            # --- TOTAL LOSS ---
            loss_total = lambda_simclr * loss_simclr + lambda_clip * loss_clip
            
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            losses_simclr.append(loss_simclr.item())
            losses_clip.append(loss_clip.item())
        
        sim_val, clip_val, sum_val = compute_validation_losses_new(val_loader, model, transform)

        print(f"[Epoch {epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
        losses_simclr_all_epoch.append(np.mean(losses_simclr))
        losses_clip_all_epoch.append(np.mean(losses_clip))
        losses_simclr_val_all_epoch.append(sim_val)
        losses_clip_val_all_epoch.append(clip_val)
        losses_sum_val_all_epoch.append(sum_val)

        torch.save({
            "clip": model.state_dict(),
        }, weightPath+f"SLIP_joint_{name}/{epoch}.pth")
    

    plt.plot(losses_simclr_all_epoch, label="InfoNCE Loss", color='b')
    plt.plot(losses_clip_all_epoch, label="CLIP Loss", color='g')
    plt.plot(losses_simclr_val_all_epoch, label="Val InfoNCE", color='r')
    plt.plot(losses_clip_val_all_epoch,   label="Val CLIP", color='orange')
    plt.plot(losses_sum_val_all_epoch,   label="Val CLIP+InfoNCE", color='yellow')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("SLIP Training: SimCLR and CLIP")
    plt.legend()
    plt.savefig(lossPath+f"loss_slip_joint_{name}.png")
    plt.close()





def train_simclr_then_clip(train_loader, val_loader, transform, name, epochs=10, freeze_simclr=False):
    clip_model.train()
    proj.train()
    losses_simclr_all_epoch = []
    losses_clip_all_epoch = []
    losses_simclr_val_all_epoch = []
    losses_clip_val_all_epoch = []
    losses_sum_val_all_epoch = []

    if not os.path.exists(weightPath+f"SLIP_simclr_then_clip_{name}"):
        os.mkdir(weightPath+f"SLIP_simclr_then_clip_{name}")


    # --- SIMCLR PART FIRST ---

    for epoch in range(int(epochs/2)):
        losses_simclr = []
        losses_clip = []

        for (img, img_paths, captions, idx) in tqdm(train_loader):
            images = [Image.open(p).convert("RGB") for p in img_paths]
            
            # --- SIMCLR ---
            
            x = img #.to(device)
            x2 = transform(img) #.to(device)
            
            v1 = clip_model.get_image_features(pixel_values=x)
            v2 = clip_model.get_image_features(pixel_values=x2)

            z1 = proj(v1)
            z2 = proj(v2)

            loss_simclr = infoNCEloss(z1, z2, temperature)

            with torch.no_grad(): # juste pour voir comment ca evolue du cote texte du coup
                # --- CLIP ---
                
                inputs = processor(images=images, text=captions, return_tensors="pt", padding=True) #.to(device)

                outputs = clip_model(**inputs)
                img_emb = outputs.image_embeds
                txt_emb = outputs.text_embeds

                loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)

            
            optimizer.zero_grad()
            loss_simclr.backward()
            optimizer.step()

            losses_simclr.append(loss_simclr.item())
            losses_clip.append(loss_clip.item())

        # --- Validation ---
        sim_val, clip_val, sum_val = compute_validation_losses(val_loader, clip_model, proj, transform)

        print(f"[Epoch {epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
        losses_simclr_all_epoch.append(np.mean(losses_simclr))
        losses_clip_all_epoch.append(np.mean(losses_clip))
        losses_simclr_val_all_epoch.append(sim_val)
        losses_clip_val_all_epoch.append(clip_val)
        losses_sum_val_all_epoch.append(sum_val)

        
        torch.save({
            "clip": clip_model.state_dict(),
            "proj": proj.state_dict()
        }, weightPath+f"SLIP_simclr_then_clip_{name}/{epoch}.pth")
    


    # --- CLIP PART SECOND ---

    if freeze_simclr:
        # Freeze all vision model parameters
        for param in clip_model.vision_model.parameters():
            param.requires_grad = False


    for epoch in range(int(epochs/2), epochs):
        losses_simclr = []
        losses_clip = []

        for (img, img_paths, captions, idx) in tqdm(train_loader):
            images = [Image.open(p).convert("RGB") for p in img_paths]
            
            with torch.no_grad():
                # --- SIMCLR  ---
                
                x = img #.to(device)
                x2 = transform(img) #.to(device)

                v1 = clip_model.get_image_features(pixel_values=x)
                v2 = clip_model.get_image_features(pixel_values=x2)

                z1 = proj(v1)
                z2 = proj(v2)

                loss_simclr = infoNCEloss(z1, z2, temperature)

            # --- CLIP ---
            
            inputs = processor(images=images, text=captions, return_tensors="pt", padding=True) #.to(device)

            outputs = clip_model(**inputs)
            
            if freeze_simclr:
                img_emb = outputs.image_embeds.detach()
            else:
                img_emb = outputs.image_embeds

            txt_emb = outputs.text_embeds

            loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)
            
            optimizer.zero_grad()
            loss_clip.backward()
            optimizer.step()

            losses_simclr.append(loss_simclr.item())
            losses_clip.append(loss_clip.item())

        # --- Validation ---
        sim_val, clip_val, sum_val = compute_validation_losses(val_loader, clip_model, proj, transform)

        print(f"[Epoch {epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
        losses_simclr_all_epoch.append(np.mean(losses_simclr))
        losses_clip_all_epoch.append(np.mean(losses_clip))
        losses_simclr_val_all_epoch.append(sim_val)
        losses_clip_val_all_epoch.append(clip_val)
        losses_sum_val_all_epoch.append(sum_val)

        torch.save({
            "clip": clip_model.state_dict(),
            "proj": proj.state_dict()
        }, weightPath+f"SLIP_simclr_then_clip_{name}/{epoch}.pth")
    

    plt.plot(losses_simclr_all_epoch, label="InfoNCE Loss", color='b')
    plt.plot(losses_clip_all_epoch, label="CLIP Loss", color='g')
    plt.plot(losses_simclr_val_all_epoch, label="Val InfoNCE", color='r')
    plt.plot(losses_clip_val_all_epoch,   label="Val CLIP", color='orange')
    plt.plot(losses_sum_val_all_epoch,   label="Val CLIP+InfoNCE", color='yellow')
    plt.axvline(x=int(epochs/2), color='r', linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("SLIP Training: SimCLR then CLIP")
    plt.legend()
    plt.savefig(lossPath+f"loss_slip_simclr_then_clip_{name}.png")
    plt.close()





def train_simclr_then_clip_new(train_loader, val_loader, transform, name, epochs=10, freeze_simclr=False):
    model.train()
    losses_simclr_all_epoch = []
    losses_clip_all_epoch = []
    losses_simclr_val_all_epoch = []
    losses_clip_val_all_epoch = []
    losses_sum_val_all_epoch = []

    if not os.path.exists(weightPath+f"SLIP_simclr_then_clip_{name}"):
        os.mkdir(weightPath+f"SLIP_simclr_then_clip_{name}")


    # --- SIMCLR PART FIRST ---

    for epoch in range(int(epochs/2)):
        losses_simclr = []
        losses_clip = []

        for (img, img_paths, captions, idx) in tqdm(train_loader):
            
            # --- SIMCLR ---
            
            x = img #.to(device)
            x2 = transform(img) #.to(device)
            
            v1 = model.encode_image(x)
            v2 = model.encode_image(x2)

            loss_simclr = infoNCEloss(v1, v2, temperature)

            with torch.no_grad(): # juste pour voir comment ca evolue du cote texte du coup
                # --- CLIP ---
                
                text = clip.tokenize(list(captions))#.to(device)
                img_emb, txt_emb = model(img, text)

                loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)

            
            optimizer.zero_grad()
            loss_simclr.backward()
            optimizer.step()

            losses_simclr.append(loss_simclr.item())
            losses_clip.append(loss_clip.item())

        # --- Validation ---
        sim_val, clip_val, sum_val = compute_validation_losses_new(val_loader, model, transform)

        print(f"[Epoch {epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
        losses_simclr_all_epoch.append(np.mean(losses_simclr))
        losses_clip_all_epoch.append(np.mean(losses_clip))
        losses_simclr_val_all_epoch.append(sim_val)
        losses_clip_val_all_epoch.append(clip_val)
        losses_sum_val_all_epoch.append(sum_val)
        
        torch.save({
            "clip": model.state_dict(),
        }, weightPath+f"SLIP_simclr_then_clip_{name}/{epoch}.pth")
    

    # --- CLIP PART SECOND ---

    if freeze_simclr:
        # Freeze all vision model parameters
        for param in model.visual.parameters():
            param.requires_grad = False


    for epoch in range(int(epochs/2), epochs):
        losses_simclr = []
        losses_clip = []

        for (img, img_paths, captions, idx) in tqdm(train_loader):
            
            with torch.no_grad():
                # --- SIMCLR  ---
                
                x = img #.to(device)
                x2 = transform(img) #.to(device)
                v1 = model.encode_image(x)
                v2 = model.encode_image(x2)

                loss_simclr = infoNCEloss(v1, v2, temperature)


            # --- CLIP ---
            
            text = clip.tokenize(list(captions))#.to(device)
            img_emb, txt_emb = model(img, text)
            
            if freeze_simclr:
                img_emb = img_emb.detach()


            loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)
            
            optimizer.zero_grad()
            loss_clip.backward()
            optimizer.step()

            losses_simclr.append(loss_simclr.item())
            losses_clip.append(loss_clip.item())

        # --- Validation ---
        sim_val, clip_val, sum_val = compute_validation_losses_new(val_loader, model, transform)

        print(f"[Epoch {epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
        losses_simclr_all_epoch.append(np.mean(losses_simclr))
        losses_clip_all_epoch.append(np.mean(losses_clip))
        losses_simclr_val_all_epoch.append(sim_val)
        losses_clip_val_all_epoch.append(clip_val)
        losses_sum_val_all_epoch.append(sum_val)

        torch.save({
            "clip": model.state_dict(),
        }, weightPath+f"SLIP_simclr_then_clip_{name}/{epoch}.pth")
    

    plt.plot(losses_simclr_all_epoch, label="InfoNCE Loss", color='b')
    plt.plot(losses_clip_all_epoch, label="CLIP Loss", color='g')
    plt.plot(losses_simclr_val_all_epoch, label="Val InfoNCE", color='r')
    plt.plot(losses_clip_val_all_epoch,   label="Val CLIP", color='orange')
    plt.plot(losses_sum_val_all_epoch,   label="Val CLIP+InfoNCE", color='yellow')
    plt.axvline(x=int(epochs/2), color='r', linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("SLIP Training: SimCLR then CLIP")
    plt.legend()
    plt.savefig(lossPath+f"loss_slip_simclr_then_clip_{name}.png")
    plt.close()





def train_clip_then_simclr(train_loader, val_loader, transform, name, epochs=30):
    clip_model.train()
    proj.train()
    losses_simclr_all_epoch = []
    losses_clip_all_epoch = []
    losses_simclr_val_all_epoch = []
    losses_clip_val_all_epoch = []
    losses_sum_val_all_epoch = []

    if not os.path.exists(weightPath+f"SLIP_clip_then_simclr_{name}"):
        os.mkdir(weightPath+f"SLIP_clip_then_simclr_{name}")



    # --- CLIP PART FIRST ---

    for epoch in range(int(epochs/2)):
        losses_simclr = []
        losses_clip = []

        for (img, img_paths, captions, idx) in tqdm(train_loader):
            images = [Image.open(p).convert("RGB") for p in img_paths]

            with torch.no_grad():
                # --- SIMCLR ---
                
                x = img #.to(device)
                x2 = transform(img) #.to(device)
                
                v1 = clip_model.get_image_features(pixel_values=x)
                v2 = clip_model.get_image_features(pixel_values=x2)

                z1 = proj(v1)
                z2 = proj(v2)

                loss_simclr = infoNCEloss(z1, z2, temperature)

            # --- CLIP ---
                
            inputs = processor(images=images, text=captions, return_tensors="pt", padding=True) #.to(device)

            outputs = clip_model(**inputs)
            img_emb = outputs.image_embeds
            txt_emb = outputs.text_embeds

            loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)

            
            optimizer.zero_grad()
            loss_clip.backward()
            optimizer.step()

            losses_simclr.append(loss_simclr.item())
            losses_clip.append(loss_clip.item())

        # --- Validation ---
        sim_val, clip_val, sum_val = compute_validation_losses(val_loader, clip_model, proj, transform)

        print(f"[Epoch {epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
        losses_simclr_all_epoch.append(np.mean(losses_simclr))
        losses_clip_all_epoch.append(np.mean(losses_clip))
        losses_simclr_val_all_epoch.append(sim_val)
        losses_clip_val_all_epoch.append(clip_val)
        losses_sum_val_all_epoch.append(sum_val)

        torch.save({
            "clip": clip_model.state_dict(),
            "proj": proj.state_dict()
        }, weightPath+f"SLIP_clip_then_simclr_{name}/{epoch}.pth")
    


    # --- SIMCLR PART SECOND ---

    for epoch in range(int(epochs/2), epochs):
        losses_simclr = []
        losses_clip = []

        for (img, img_paths, captions, idx) in tqdm(train_loader):
            images = [Image.open(p).convert("RGB") for p in img_paths]
            
            
            # --- SIMCLR  ---
                
            x = img #.to(device)
            x2 = transform(img) #.to(device)

            v1 = clip_model.get_image_features(pixel_values=x)
            v2 = clip_model.get_image_features(pixel_values=x2)

            z1 = proj(v1)
            z2 = proj(v2)

            loss_simclr = infoNCEloss(z1, z2, temperature)


            with torch.no_grad():
                # --- CLIP ---
                
                inputs = processor(images=images, text=captions, return_tensors="pt", padding=True) #.to(device)

                outputs = clip_model(**inputs)
                img_emb = outputs.image_embeds
                txt_emb = outputs.text_embeds

                loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)
            
            optimizer.zero_grad()
            loss_simclr.backward()
            optimizer.step()

            losses_simclr.append(loss_simclr.item())
            losses_clip.append(loss_clip.item())

        # --- Validation ---
        sim_val, clip_val, sum_val = compute_validation_losses(val_loader, clip_model, proj, transform)
        print(f"[Epoch {epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
        losses_simclr_all_epoch.append(np.mean(losses_simclr))
        losses_clip_all_epoch.append(np.mean(losses_clip))
        losses_simclr_val_all_epoch.append(sim_val)
        losses_clip_val_all_epoch.append(clip_val)
        losses_sum_val_all_epoch.append(sum_val)

        torch.save({
            "clip": clip_model.state_dict(),
            "proj": proj.state_dict()
        }, weightPath+f"SLIP_clip_then_simclr_{name}/{epoch}.pth")
    

    plt.plot(losses_simclr_all_epoch, label="InfoNCE Loss", color='b')
    plt.plot(losses_clip_all_epoch, label="CLIP Loss", color='g')
    plt.plot(losses_simclr_val_all_epoch, label="Val InfoNCE", color='r')
    plt.plot(losses_clip_val_all_epoch,   label="Val CLIP", color='orange')
    plt.plot(losses_sum_val_all_epoch,   label="Val CLIP+InfoNCE", color='yellow')
    plt.axvline(x=int(epochs/2), color='r', linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("SLIP Training: CLIP then SimCLR")
    plt.legend()
    plt.savefig(lossPath+f"loss_slip_clip_then_simclr_{name}.png")
    plt.close()



def train_clip_then_simclr_new(train_loader, val_loader, transform, name, epochs=30):
    model.train()
    losses_simclr_all_epoch = []
    losses_clip_all_epoch = []
    losses_simclr_val_all_epoch = []
    losses_clip_val_all_epoch = []
    losses_sum_val_all_epoch = []

    if not os.path.exists(weightPath+f"SLIP_clip_then_simclr_{name}"):
        os.mkdir(weightPath+f"SLIP_clip_then_simclr_{name}")



    # --- CLIP PART FIRST ---

    for epoch in range(int(epochs/2)):
        losses_simclr = []
        losses_clip = []

        for (img, img_paths, captions, idx) in tqdm(train_loader):

            with torch.no_grad():
                # --- SIMCLR ---
                
                x = img #.to(device)
                x2 = transform(img) #.to(device)
                
                v1 = model.encode_image(x)
                v2 = model.encode_image(x2)

                loss_simclr = infoNCEloss(v1, v2, temperature)

            # --- CLIP ---
                
            text = clip.tokenize(list(captions))#.to(device)
            img_emb, txt_emb = model(img, text)

            loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)

            
            optimizer.zero_grad()
            loss_clip.backward()
            optimizer.step()

            losses_simclr.append(loss_simclr.item())
            losses_clip.append(loss_clip.item())

        # --- Validation ---
        sim_val, clip_val, sum_val = compute_validation_losses_new(val_loader, model, transform)

        print(f"[Epoch {epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
        losses_simclr_all_epoch.append(np.mean(losses_simclr))
        losses_clip_all_epoch.append(np.mean(losses_clip))
        losses_simclr_val_all_epoch.append(sim_val)
        losses_clip_val_all_epoch.append(clip_val)
        losses_sum_val_all_epoch.append(sum_val)

        torch.save({
            "clip": model.state_dict(),
        }, weightPath+f"SLIP_clip_then_simclr_{name}/{epoch}.pth")
    


    # --- SIMCLR PART SECOND ---

    for epoch in range(int(epochs/2), epochs):
        losses_simclr = []
        losses_clip = []

        for (img, img_paths, captions, idx) in tqdm(train_loader):
            images = [Image.open(p).convert("RGB") for p in img_paths]
            
            
            # --- SIMCLR  ---
                
            x = img #.to(device)
            x2 = transform(img) #.to(device)

            v1 = model.encode_image(x)
            v2 = model.encode_image(x2)

            loss_simclr = infoNCEloss(v1, v2, temperature)


            with torch.no_grad():
                # --- CLIP ---
                
                text = clip.tokenize(list(captions))#.to(device)
                img_emb, txt_emb = model(img, text)

                loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)
            
            optimizer.zero_grad()
            loss_simclr.backward()
            optimizer.step()

            losses_simclr.append(loss_simclr.item())
            losses_clip.append(loss_clip.item())

        # --- Validation ---
        sim_val, clip_val, sum_val = compute_validation_losses_new(val_loader, model, transform)
        print(f"[Epoch {epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
        losses_simclr_all_epoch.append(np.mean(losses_simclr))
        losses_clip_all_epoch.append(np.mean(losses_clip))
        losses_simclr_val_all_epoch.append(sim_val)
        losses_clip_val_all_epoch.append(clip_val)
        losses_sum_val_all_epoch.append(sum_val)

        torch.save({
            "clip": model.state_dict(),
        }, weightPath+f"SLIP_clip_then_simclr_{name}/{epoch}.pth")
    

    plt.plot(losses_simclr_all_epoch, label="InfoNCE Loss", color='b')
    plt.plot(losses_clip_all_epoch, label="CLIP Loss", color='g')
    plt.plot(losses_simclr_val_all_epoch, label="Val InfoNCE", color='r')
    plt.plot(losses_clip_val_all_epoch,   label="Val CLIP", color='orange')
    plt.plot(losses_sum_val_all_epoch,   label="Val CLIP+InfoNCE", color='yellow')
    plt.axvline(x=int(epochs/2), color='r', linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("SLIP Training: CLIP then SimCLR")
    plt.legend()
    plt.savefig(lossPath+f"loss_slip_clip_then_simclr_{name}.png")
    plt.close()



def train_simclr_then_clip_then_both(train_loader, val_loader, transform, name, epochs=30, lambda_clip = 1.0, lambda_simclr = 1.0):
    clip_model.train()
    proj.train()
    losses_simclr_all_epoch = []
    losses_clip_all_epoch = []
    losses_simclr_val_all_epoch = []
    losses_clip_val_all_epoch = []
    losses_sum_val_all_epoch = []
    
    if not os.path.exists(weightPath+f"SLIP_simclr_then_clip_then_both_{name}"):
        os.mkdir(weightPath+f"SLIP_simclr_then_clip_then_both_{name}")


    # --- SIMCLR PART FIRST ---

    for epoch in range(int(epochs/3)):
        losses_simclr = []
        losses_clip = []

        for (img, img_paths, captions, idx) in tqdm(train_loader):
            images = [Image.open(p).convert("RGB") for p in img_paths]
            
            # --- SIMCLR ---
            
            x = img #.to(device)
            x2 = transform(img) #.to(device)
            
            v1 = clip_model.get_image_features(pixel_values=x)
            v2 = clip_model.get_image_features(pixel_values=x2)

            z1 = proj(v1)
            z2 = proj(v2)

            loss_simclr = infoNCEloss(z1, z2, temperature)

            with torch.no_grad(): # juste pour voir comment ca evolue du cote texte du coup
                # --- CLIP ---
                
                inputs = processor(images=images, text=captions, return_tensors="pt", padding=True) #.to(device)

                outputs = clip_model(**inputs)
                img_emb = outputs.image_embeds
                txt_emb = outputs.text_embeds

                loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)

            
            optimizer.zero_grad()
            loss_simclr.backward()
            optimizer.step()

            losses_simclr.append(loss_simclr.item())
            losses_clip.append(loss_clip.item())

        # --- Validation ---
        sim_val, clip_val, sum_val = compute_validation_losses(val_loader, clip_model, proj, transform)

        print(f"[Epoch {epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
        losses_simclr_all_epoch.append(np.mean(losses_simclr))
        losses_clip_all_epoch.append(np.mean(losses_clip))
        losses_simclr_val_all_epoch.append(sim_val)
        losses_clip_val_all_epoch.append(clip_val)
        losses_sum_val_all_epoch.append(sum_val)

        
        torch.save({
            "clip": clip_model.state_dict(),
            "proj": proj.state_dict()
        }, weightPath+f"SLIP_simclr_then_clip_then_both_{name}/{epoch}.pth")
    


    # --- CLIP PART SECOND ---

    for epoch in range(int(epochs/3), int(2*epochs/3)):
        losses_simclr = []
        losses_clip = []

        for (img, img_paths, captions, idx) in tqdm(train_loader):
            images = [Image.open(p).convert("RGB") for p in img_paths]
            
            with torch.no_grad():
                # --- SIMCLR  ---
                
                x = img #.to(device)
                x2 = transform(img) #.to(device)

                v1 = clip_model.get_image_features(pixel_values=x)
                v2 = clip_model.get_image_features(pixel_values=x2)

                z1 = proj(v1)
                z2 = proj(v2)

                loss_simclr = infoNCEloss(z1, z2, temperature)

            # --- CLIP ---
            
            inputs = processor(images=images, text=captions, return_tensors="pt", padding=True) #.to(device)

            outputs = clip_model(**inputs)
            
           
            img_emb = outputs.image_embeds

            txt_emb = outputs.text_embeds

            loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)
            
            optimizer.zero_grad()
            loss_clip.backward()
            optimizer.step()

            losses_simclr.append(loss_simclr.item())
            losses_clip.append(loss_clip.item())

        # --- Validation ---
        sim_val, clip_val, sum_val = compute_validation_losses(val_loader, clip_model, proj, transform)

        print(f"[Epoch {epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
        losses_simclr_all_epoch.append(np.mean(losses_simclr))
        losses_clip_all_epoch.append(np.mean(losses_clip))
        losses_simclr_val_all_epoch.append(sim_val)
        losses_clip_val_all_epoch.append(clip_val)
        losses_sum_val_all_epoch.append(sum_val)

        torch.save({
            "clip": clip_model.state_dict(),
            "proj": proj.state_dict()
        }, weightPath+f"SLIP_simclr_then_clip_then_both_{name}/{epoch}.pth")

    

    # JOINT PART
    for epoch in range(int(2*epochs/3),epochs):
        losses_simclr = []
        losses_clip = []

        for (img, img_paths, captions, idx) in tqdm(train_loader):
            images = [Image.open(p).convert("RGB") for p in img_paths]
            
            # --- SIMCLR PART ---
            
            x = img #.to(device)
            x2 = transform(img) #.to(device)

            v1 = clip_model.get_image_features(pixel_values=x)
            v2 = clip_model.get_image_features(pixel_values=x2)

            z1 = proj(v1)
            z2 = proj(v2)

            loss_simclr = infoNCEloss(z1, z2, temperature)

            # --- CLIP PART ---
            
            inputs = processor(images=images, text=captions, return_tensors="pt", padding=True) #.to(device)

            outputs = clip_model(**inputs)
            img_emb = outputs.image_embeds
            txt_emb = outputs.text_embeds

            loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)

            # --- TOTAL LOSS ---
            loss_total = lambda_simclr * loss_simclr + lambda_clip * loss_clip
            
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            losses_simclr.append(loss_simclr.item())
            losses_clip.append(loss_clip.item())

        # --- Validation ---
        sim_val, clip_val, sum_val = compute_validation_losses(val_loader, clip_model, proj, transform)

        print(f"[Epoch {epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
        losses_simclr_all_epoch.append(np.mean(losses_simclr))
        losses_clip_all_epoch.append(np.mean(losses_clip))
        losses_simclr_val_all_epoch.append(sim_val)
        losses_clip_val_all_epoch.append(clip_val)
        losses_sum_val_all_epoch.append(sum_val)

        torch.save({
            "clip": clip_model.state_dict(),
            "proj": proj.state_dict()
        }, weightPath+f"SLIP_simclr_then_clip_then_both_{name}/{epoch}.pth")
    

    plt.plot(losses_simclr_all_epoch, label="InfoNCE Loss", color='b')
    plt.plot(losses_clip_all_epoch, label="CLIP Loss", color='g')
    plt.plot(losses_simclr_val_all_epoch, label="Val InfoNCE", color='r')
    plt.plot(losses_clip_val_all_epoch,   label="Val CLIP", color='orange')
    plt.plot(losses_sum_val_all_epoch,   label="Val CLIP+InfoNCE", color='yellow')
    plt.axvline(x=int(epochs/3), color='r', linestyle='--')
    plt.axvline(x=int(2*epochs/3), color='r', linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("SLIP Training: SimCLR then CLIP then both")
    plt.legend()
    plt.savefig(lossPath+f"loss_slip_simclr_then_clip_then_both_{name}.png")
    plt.close()




def train_simclr_then_clip_then_both_new(train_loader, val_loader, transform, name, epochs=30, lambda_clip = 1.0, lambda_simclr = 1.0):
    model.train()
    losses_simclr_all_epoch = []
    losses_clip_all_epoch = []
    losses_simclr_val_all_epoch = []
    losses_clip_val_all_epoch = []
    losses_sum_val_all_epoch = []
    
    if not os.path.exists(weightPath+f"SLIP_simclr_then_clip_then_both_{name}"):
        os.mkdir(weightPath+f"SLIP_simclr_then_clip_then_both_{name}")


    # --- SIMCLR PART FIRST ---

    for epoch in range(int(epochs/3)):
        losses_simclr = []
        losses_clip = []

        for (img, img_paths, captions, idx) in tqdm(train_loader):
            
            # --- SIMCLR ---
            
            x = img #.to(device)
            x2 = transform(img) #.to(device)
            
            v1 = model.encode_image(x)
            v2 = model.encode_image(x2)

            loss_simclr = infoNCEloss(v1, v2, temperature)

            with torch.no_grad(): # juste pour voir comment ca evolue du cote texte du coup
                # --- CLIP ---
                
                text = clip.tokenize(list(captions))#.to(device)
                img_emb, txt_emb = model(img, text)

                loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)

            
            optimizer.zero_grad()
            loss_simclr.backward()
            optimizer.step()

            losses_simclr.append(loss_simclr.item())
            losses_clip.append(loss_clip.item())

        # --- Validation ---
        sim_val, clip_val, sum_val = compute_validation_losses_new(val_loader, model, transform)

        print(f"[Epoch {epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
        losses_simclr_all_epoch.append(np.mean(losses_simclr))
        losses_clip_all_epoch.append(np.mean(losses_clip))
        losses_simclr_val_all_epoch.append(sim_val)
        losses_clip_val_all_epoch.append(clip_val)
        losses_sum_val_all_epoch.append(sum_val)

        
        torch.save({
            "clip": model.state_dict(),
        }, weightPath+f"SLIP_simclr_then_clip_then_both_{name}/{epoch}.pth")
    


    # --- CLIP PART SECOND ---

    for epoch in range(int(epochs/3), int(2*epochs/3)):
        losses_simclr = []
        losses_clip = []

        for (img, img_paths, captions, idx) in tqdm(train_loader):
            
            with torch.no_grad():
                # --- SIMCLR  ---
                
                x = img #.to(device)
                x2 = transform(img) #.to(device)

                v1 = model.encode_image(x)
                v2 = model.encode_image(x2)

                loss_simclr = infoNCEloss(v1, v2, temperature)

            # --- CLIP ---
            
            text = clip.tokenize(list(captions))#.to(device)
            img_emb, txt_emb = model(img, text)

            loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)
            
            optimizer.zero_grad()
            loss_clip.backward()
            optimizer.step()

            losses_simclr.append(loss_simclr.item())
            losses_clip.append(loss_clip.item())

        # --- Validation ---
        sim_val, clip_val, sum_val = compute_validation_losses_new(val_loader, model, transform)

        print(f"[Epoch {epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
        losses_simclr_all_epoch.append(np.mean(losses_simclr))
        losses_clip_all_epoch.append(np.mean(losses_clip))
        losses_simclr_val_all_epoch.append(sim_val)
        losses_clip_val_all_epoch.append(clip_val)
        losses_sum_val_all_epoch.append(sum_val)

        torch.save({
            "clip": model.state_dict(),
        }, weightPath+f"SLIP_simclr_then_clip_then_both_{name}/{epoch}.pth")

    

    # JOINT PART
    for epoch in range(int(2*epochs/3),epochs):
        losses_simclr = []
        losses_clip = []

        for (img, img_paths, captions, idx) in tqdm(train_loader):
            
            # --- SIMCLR PART ---
            
            x = img #.to(device)
            x2 = transform(img) #.to(device)

            v1 = model.encode_image(x)
            v2 = model.encode_image(x2)

            loss_simclr = infoNCEloss(v1, v2, temperature)

            # --- CLIP PART ---
            
            text = clip.tokenize(list(captions))#.to(device)
            img_emb, txt_emb = model(img, text)

            loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)

            # --- TOTAL LOSS ---
            loss_total = lambda_simclr * loss_simclr + lambda_clip * loss_clip
            
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            losses_simclr.append(loss_simclr.item())
            losses_clip.append(loss_clip.item())

        # --- Validation ---
        sim_val, clip_val, sum_val = compute_validation_losses_new(val_loader, model, transform)

        print(f"[Epoch {epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
        losses_simclr_all_epoch.append(np.mean(losses_simclr))
        losses_clip_all_epoch.append(np.mean(losses_clip))
        losses_simclr_val_all_epoch.append(sim_val)
        losses_clip_val_all_epoch.append(clip_val)
        losses_sum_val_all_epoch.append(sum_val)

        torch.save({
            "clip": model.state_dict(),
        }, weightPath+f"SLIP_simclr_then_clip_then_both_{name}/{epoch}.pth")
    

    plt.plot(losses_simclr_all_epoch, label="InfoNCE Loss", color='b')
    plt.plot(losses_clip_all_epoch, label="CLIP Loss", color='g')
    plt.plot(losses_simclr_val_all_epoch, label="Val InfoNCE", color='r')
    plt.plot(losses_clip_val_all_epoch,   label="Val CLIP", color='orange')
    plt.plot(losses_sum_val_all_epoch,   label="Val CLIP+InfoNCE", color='yellow')
    plt.axvline(x=int(epochs/3), color='r', linestyle='--')
    plt.axvline(x=int(2*epochs/3), color='r', linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("SLIP Training: SimCLR then CLIP then both")
    plt.legend()
    plt.savefig(lossPath+f"loss_slip_simclr_then_clip_then_both_{name}.png")
    plt.close()




def train_clip_with_degrad(train_loader, val_loader, transform, name, epochs=20, nb_dupli_degrad=3):
    clip_model.train()
    proj.train()
    losses_clip_all_epoch = []
    losses_clip_val_all_epoch = []

    if not os.path.exists(weightPath+f"SLIP_clip_with_degrad_{name}"):
        os.mkdir(weightPath+f"SLIP_clip_with_degrad_{name}")



    # --- CLIP PART FIRST ---

    for epoch in range(epochs):
        losses_clip = []
        val_losses_clip = []

        for (img, img_paths, captions, idx) in tqdm(train_loader):
            #images = [Image.open(p).convert("RGB") for p in img_paths]

            x = img #.to(device)
            inputs = processor(images=x, text=captions, return_tensors="pt", padding=True, do_rescale=False) #.to(device)

            outputs = clip_model(**inputs)
            img_emb = outputs.image_embeds
            txt_emb = outputs.text_embeds

            loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)

            
            optimizer.zero_grad()
            loss_clip.backward()
            optimizer.step()

            losses_clip.append(loss_clip.item())

            for _ in range(nb_dupli_degrad):
                x2 = transform(img).clamp(0, 1) #.to(device)
                
                inputs = processor(images=x2, text=captions, return_tensors="pt", padding=True, do_rescale=False) #.to(device)

                outputs = clip_model(**inputs)
                img_emb = outputs.image_embeds
                txt_emb = outputs.text_embeds

                loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)

                
                optimizer.zero_grad()
                loss_clip.backward()
                optimizer.step()

                losses_clip.append(loss_clip.item())

 
        # --- Validation ---
        sim_val, clip_val, sum_val = compute_validation_losses(val_loader, clip_model, proj, transform)

        print(f"[Epoch {epoch}] CLIP: {np.mean(losses_clip):.4f}")
        losses_clip_all_epoch.append(np.mean(losses_clip))
        losses_clip_val_all_epoch.append(clip_val)

        torch.save({
            "clip": clip_model.state_dict(),
            "proj": proj.state_dict()
        }, weightPath+f"SLIP_clip_with_degrad_{name}/{epoch}.pth")
    

    plt.plot(losses_clip_all_epoch, label="CLIP Loss (train)", color='g')
    plt.plot(losses_clip_val_all_epoch, label="CLIP Loss (val)", color='r')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("CLIP with degradations")
    plt.legend()
    plt.savefig(lossPath+f"loss_slip_clip_with_degrad_{name}.png")
    plt.close()



def train_clip_with_degrad_new(train_loader, val_loader, transform, name, epochs=20, nb_dupli_degrad=3):
    model.train()
    losses_clip_all_epoch = []
    losses_clip_val_all_epoch = []

    if not os.path.exists(weightPath+f"SLIP_clip_with_degrad_{name}"):
        os.mkdir(weightPath+f"SLIP_clip_with_degrad_{name}")



    for epoch in range(epochs):
        losses_clip = []

        for (img, img_paths, captions, idx) in tqdm(train_loader):

            text = clip.tokenize(list(captions))#.to(device)
            img_emb, txt_emb = model(img, text)

            loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)

            optimizer.zero_grad()
            loss_clip.backward()
            optimizer.step()

            losses_clip.append(loss_clip.item())

            for _ in range(nb_dupli_degrad):
                x2 = transform(img).clamp(0, 1) #.to(device)
                
                text = clip.tokenize(list(captions))#.to(device)
                img_emb, txt_emb = model(x2, text)

                loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)

                
                optimizer.zero_grad()
                loss_clip.backward()
                optimizer.step()

                losses_clip.append(loss_clip.item())

 
        # --- Validation ---
        sim_val, clip_val, sum_val = compute_validation_losses_new(val_loader, model, transform)

        print(f"[Epoch {epoch}] CLIP: {np.mean(losses_clip):.4f}")
        losses_clip_all_epoch.append(np.mean(losses_clip))
        losses_clip_val_all_epoch.append(clip_val)

        torch.save({
            "clip": model.state_dict()
        }, weightPath+f"SLIP_clip_with_degrad_{name}/{epoch}.pth")
    

    plt.plot(losses_clip_all_epoch, label="CLIP Loss (train)", color='g')
    plt.plot(losses_clip_val_all_epoch, label="CLIP Loss (val)", color='r')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("CLIP with degradations")
    plt.legend()
    plt.savefig(lossPath+f"loss_slip_clip_with_degrad_{name}.png")
    plt.close()



def train_simclr_only(train_loader, val_loader, transform, name, epochs=10):
    clip_model.train()
    proj.train()
    losses_simclr_all_epoch = []
    losses_clip_all_epoch = []
    losses_simclr_val_all_epoch = []
    losses_clip_val_all_epoch = []
    losses_sum_val_all_epoch = []

    if not os.path.exists(weightPath+f"SLIP_simclr_only_{name}"):
        os.mkdir(weightPath+f"SLIP_simclr_only_{name}")


    # --- SIMCLR PART  ---

    for epoch in range(epochs):
        losses_simclr = []
        losses_clip = []

        for (img, img_paths, captions, idx) in tqdm(train_loader):
            images = [Image.open(p).convert("RGB") for p in img_paths]
            
            # --- SIMCLR ---
            
            x = img #.to(device)
            x2 = transform(img) #.to(device)
            
            v1 = clip_model.get_image_features(pixel_values=x)
            v2 = clip_model.get_image_features(pixel_values=x2)

            z1 = proj(v1)
            z2 = proj(v2)

            loss_simclr = infoNCEloss(z1, z2, temperature)

            with torch.no_grad(): # juste pour voir comment ca evolue du cote texte du coup
                # --- CLIP ---
                
                inputs = processor(images=images, text=captions, return_tensors="pt", padding=True) #.to(device)

                outputs = clip_model(**inputs)
                img_emb = outputs.image_embeds
                txt_emb = outputs.text_embeds

                loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)

            
            optimizer.zero_grad()
            loss_simclr.backward()
            optimizer.step()

            losses_simclr.append(loss_simclr.item())
            losses_clip.append(loss_clip.item())

        # --- Validation ---
        sim_val, clip_val, sum_val = compute_validation_losses(val_loader, clip_model, proj, transform)

        print(f"[Epoch {epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
        losses_simclr_all_epoch.append(np.mean(losses_simclr))
        losses_clip_all_epoch.append(np.mean(losses_clip))
        losses_simclr_val_all_epoch.append(sim_val)
        losses_clip_val_all_epoch.append(clip_val)
        losses_sum_val_all_epoch.append(sum_val)

        
        torch.save({
            "clip": clip_model.state_dict(),
            "proj": proj.state_dict()
        }, weightPath+f"SLIP_simclr_only_{name}/{epoch}.pth")
    


    plt.plot(losses_simclr_all_epoch, label="InfoNCE Loss", color='b')
    plt.plot(losses_simclr_val_all_epoch, label="Val InfoNCE", color='r')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("SLIP Training: SimCLR only")
    plt.legend()
    plt.savefig(lossPath+f"loss_slip_simclr_only_{name}.png")
    plt.close()


def train_simclr_only_new(train_loader, val_loader, transform, name, epochs=10):
    model.train()
    losses_simclr_all_epoch = []
    losses_clip_all_epoch = []
    losses_simclr_val_all_epoch = []
    losses_clip_val_all_epoch = []
    losses_sum_val_all_epoch = []

    if not os.path.exists(weightPath+f"SLIP_simclr_only_{name}"):
        os.mkdir(weightPath+f"SLIP_simclr_only_{name}")


    # --- SIMCLR PART  ---

    for epoch in range(epochs):
        losses_simclr = []
        losses_clip = []

        for (img, img_paths, captions, idx) in tqdm(train_loader):
            
            # --- SIMCLR ---
            
            x = img #.to(device)
            x2 = transform(img) #.to(device)
            
            v1 = model.encode_image(x)
            v2 = model.encode_image(x2)

            loss_simclr = infoNCEloss(v1, v2, temperature)

            with torch.no_grad(): # juste pour voir comment ca evolue du cote texte du coup
                # --- CLIP ---
                
                text = clip.tokenize(list(captions))#.to(device)
                img_emb, txt_emb = model(img, text)

                loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)

            
            optimizer.zero_grad()
            loss_simclr.backward()
            optimizer.step()

            losses_simclr.append(loss_simclr.item())
            losses_clip.append(loss_clip.item())

        # --- Validation ---
        sim_val, clip_val, sum_val = compute_validation_losses_new(val_loader, model, transform)

        print(f"[Epoch {epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
        losses_simclr_all_epoch.append(np.mean(losses_simclr))
        losses_clip_all_epoch.append(np.mean(losses_clip))
        losses_simclr_val_all_epoch.append(sim_val)
        losses_clip_val_all_epoch.append(clip_val)
        losses_sum_val_all_epoch.append(sum_val)

        
        torch.save({
            "clip": model.state_dict()
        }, weightPath+f"SLIP_simclr_only_{name}/{epoch}.pth")
    


    plt.plot(losses_simclr_all_epoch, label="InfoNCE Loss", color='b')
    plt.plot(losses_simclr_val_all_epoch, label="Val InfoNCE", color='r')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("SLIP Training: SimCLR only")
    plt.legend()
    plt.savefig(lossPath+f"loss_slip_simclr_only_{name}.png")
    plt.close()




def train_alternate_clip_simclr(train_loader, val_loader, transform, name, epochs=10, step_alternate=5):
    clip_model.train()
    proj.train()
    losses_simclr_all_epoch = []
    losses_clip_all_epoch = []
    losses_simclr_val_all_epoch = []
    losses_clip_val_all_epoch = []
    losses_sum_val_all_epoch = []

    if not os.path.exists(weightPath+f"SLIP_alternate_clip_simclr_{name}"):
        os.mkdir(weightPath+f"SLIP_alternate_clip_simclr_{name}")


    for epoch in range(int(epochs/(2*step_alternate))):
        

        for i in range(step_alternate):
            losses_simclr = []
            losses_clip = []

            for (img, img_paths, captions, idx) in tqdm(train_loader):
                images = [Image.open(p).convert("RGB") for p in img_paths]


                with torch.no_grad():
                
                    # --- SIMCLR ---
                    
                    x = img #.to(device)
                    x2 = transform(img) #.to(device)
                    
                    v1 = clip_model.get_image_features(pixel_values=x)
                    v2 = clip_model.get_image_features(pixel_values=x2)

                    z1 = proj(v1)
                    z2 = proj(v2)

                    loss_simclr = infoNCEloss(z1, z2, temperature)

                # --- CLIP ---
                    
                inputs = processor(images=images, text=captions, return_tensors="pt", padding=True) #.to(device)

                outputs = clip_model(**inputs) 
                img_emb = outputs.image_embeds
                txt_emb = outputs.text_embeds

                loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)

                
                optimizer.zero_grad()
                loss_clip.backward()
                optimizer.step()

                losses_simclr.append(loss_simclr.item())
                losses_clip.append(loss_clip.item())
            
            # --- Validation ---
            sim_val, clip_val, sum_val = compute_validation_losses(val_loader, clip_model, proj, transform)

            print(f"[Epoch {i+step_alternate*2*epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
            losses_simclr_all_epoch.append(np.mean(losses_simclr))
            losses_clip_all_epoch.append(np.mean(losses_clip))
            losses_simclr_val_all_epoch.append(sim_val)
            losses_clip_val_all_epoch.append(clip_val)
            losses_sum_val_all_epoch.append(sum_val)

            torch.save({
                "clip": clip_model.state_dict(),
                "proj": proj.state_dict()
            }, weightPath+f"SLIP_alternate_clip_simclr_{name}/{i+step_alternate*2*epoch}.pth")
    


        # --- SIMCLR PART ---

        for i in range(step_alternate):
            losses_simclr = []
            losses_clip = []
    

            for (img, img_paths, captions, idx) in tqdm(train_loader):
                images = [Image.open(p).convert("RGB") for p in img_paths]
                
                
                # --- SIMCLR  ---
                    
                x = img #.to(device)
                x2 = transform(img) #.to(device)

                v1 = clip_model.get_image_features(pixel_values=x)
                v2 = clip_model.get_image_features(pixel_values=x2)

                z1 = proj(v1)
                z2 = proj(v2)

                loss_simclr = infoNCEloss(z1, z2, temperature)


                with torch.no_grad():
                    # --- CLIP ---
                    
                    inputs = processor(images=images, text=captions, return_tensors="pt", padding=True) #.to(device)

                    outputs = clip_model(**inputs)
                    img_emb = outputs.image_embeds
                    txt_emb = outputs.text_embeds

                    loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)
                
                optimizer.zero_grad()
                loss_simclr.backward()
                optimizer.step()

                losses_simclr.append(loss_simclr.item())
                losses_clip.append(loss_clip.item())

            # --- Validation ---
            sim_val, clip_val = compute_validation_losses(val_loader, clip_model, proj, transform)

            print(f"[Epoch {i+step_alternate*2*epoch+step_alternate}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
            losses_simclr_all_epoch.append(np.mean(losses_simclr))
            losses_clip_all_epoch.append(np.mean(losses_clip))
            losses_simclr_val_all_epoch.append(sim_val)
            losses_clip_val_all_epoch.append(clip_val)

            torch.save({
                "clip": clip_model.state_dict(),
                "proj": proj.state_dict()
            }, weightPath+f"SLIP_alternate_clip_simclr_{name}/{i+step_alternate*2*epoch+step_alternate}.pth")
        

    plt.plot(losses_simclr_all_epoch, label="InfoNCE Loss", color='b')
    plt.plot(losses_clip_all_epoch, label="CLIP Loss", color='g')
    plt.plot(losses_simclr_val_all_epoch, label="Val InfoNCE", color='r')
    plt.plot(losses_clip_val_all_epoch,   label="Val CLIP", color='orange')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("SLIP Training: alternate CLIP and SimCLR")
    plt.legend()
    plt.savefig(lossPath+f"loss_slip_alternate_clip_simclr_{name}.png")
    plt.close()




def train_alternate_clip_simclr_new(train_loader, val_loader, transform, name, epochs=10, step_alternate=5):
    model.train()
    losses_simclr_all_epoch = []
    losses_clip_all_epoch = []
    losses_simclr_val_all_epoch = []
    losses_clip_val_all_epoch = []
    losses_sum_val_all_epoch = []

    if not os.path.exists(weightPath+f"SLIP_alternate_clip_simclr_{name}"):
        os.mkdir(weightPath+f"SLIP_alternate_clip_simclr_{name}")


    for epoch in range(int(epochs/(2*step_alternate))):
        

        for i in range(step_alternate):
            losses_simclr = []
            losses_clip = []

            for (img, img_paths, captions, idx) in tqdm(train_loader):
                
                with torch.no_grad():
                
                    # --- SIMCLR ---
                    
                    x = img #.to(device)
                    x2 = transform(img) #.to(device)
                    
                    v1 = model.encode_image(x)
                    v2 = model.encode_image(x2)

                    loss_simclr = infoNCEloss(v1, v2, temperature)

                # --- CLIP ---
                text = clip.tokenize(list(captions))#.to(device)
                img_emb, txt_emb = model(img, text)

                loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)
                
                optimizer.zero_grad()
                loss_clip.backward()
                optimizer.step()

                losses_simclr.append(loss_simclr.item())
                losses_clip.append(loss_clip.item())
            
            # --- Validation ---
            sim_val, clip_val, sum_val = compute_validation_losses_new(val_loader, model, transform)

            print(f"[Epoch {i+step_alternate*2*epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
            losses_simclr_all_epoch.append(np.mean(losses_simclr))
            losses_clip_all_epoch.append(np.mean(losses_clip))
            losses_simclr_val_all_epoch.append(sim_val)
            losses_clip_val_all_epoch.append(clip_val)
            losses_sum_val_all_epoch.append(sum_val)

            torch.save({
                "clip": model.state_dict(),
            }, weightPath+f"SLIP_alternate_clip_simclr_{name}/{i+step_alternate*2*epoch}.pth")
    


        # --- SIMCLR PART ---

        for i in range(step_alternate):
            losses_simclr = []
            losses_clip = []
    

            for (img, img_paths, captions, idx) in tqdm(train_loader):
                images = [Image.open(p).convert("RGB") for p in img_paths]
                
                
                # --- SIMCLR  ---
                    
                x = img #.to(device)
                x2 = transform(img) #.to(device)
                    
                v1 = model.encode_image(x)
                v2 = model.encode_image(x2)

                loss_simclr = infoNCEloss(v1, v2, temperature)



                with torch.no_grad():
                    # --- CLIP ---
                    text = clip.tokenize(list(captions))#.to(device)
                    img_emb, txt_emb = model(img, text)

                    loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)
                
                optimizer.zero_grad()
                loss_simclr.backward()
                optimizer.step()

                losses_simclr.append(loss_simclr.item())
                losses_clip.append(loss_clip.item())

            # --- Validation ---
            sim_val, clip_val, sum_val = compute_validation_losses_new(val_loader, model, transform)

            print(f"[Epoch {i+step_alternate*2*epoch+step_alternate}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
            losses_simclr_all_epoch.append(np.mean(losses_simclr))
            losses_clip_all_epoch.append(np.mean(losses_clip))
            losses_simclr_val_all_epoch.append(sim_val)
            losses_clip_val_all_epoch.append(clip_val)
            losses_sum_val_all_epoch.append(sum_val)

            torch.save({
                "clip": model.state_dict(),
            }, weightPath+f"SLIP_alternate_clip_simclr_{name}/{i+step_alternate*2*epoch+step_alternate}.pth")
        

    plt.plot(losses_simclr_all_epoch, label="InfoNCE Loss", color='b')
    plt.plot(losses_clip_all_epoch, label="CLIP Loss", color='g')
    plt.plot(losses_simclr_val_all_epoch, label="Val InfoNCE", color='r')
    plt.plot(losses_clip_val_all_epoch,   label="Val CLIP", color='orange')
    plt.plot(losses_sum_val_all_epoch,   label="Val CLIP+InfoNCE", color='yellow')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("SLIP Training: alternate CLIP and SimCLR")
    plt.legend()
    plt.savefig(lossPath+f"loss_slip_alternate_clip_simclr_{name}.png")
    plt.close()




def train_simclr_then_clip_then_both(train_loader, val_loader, transform, name, epochs=30, lambda_clip = 1.0, lambda_simclr = 1.0):
    clip_model.train()
    proj.train()
    losses_simclr_all_epoch = []
    losses_clip_all_epoch = []
    losses_simclr_val_all_epoch = []
    losses_clip_val_all_epoch = []
    losses_sum_val_all_epoch = []
    
    if not os.path.exists(weightPath+f"SLIP_simclr_then_clip_then_both_{name}"):
        os.mkdir(weightPath+f"SLIP_simclr_then_clip_then_both_{name}")


    # --- SIMCLR PART FIRST ---

    for epoch in range(int(epochs/3)):
        losses_simclr = []
        losses_clip = []

        for (img, img_paths, captions, idx) in tqdm(train_loader):
            images = [Image.open(p).convert("RGB") for p in img_paths]
            
            # --- SIMCLR ---
            
            x = img #.to(device)
            x2 = transform(img) #.to(device)
            
            v1 = clip_model.get_image_features(pixel_values=x)
            v2 = clip_model.get_image_features(pixel_values=x2)

            z1 = proj(v1)
            z2 = proj(v2)

            loss_simclr = infoNCEloss(z1, z2, temperature)

            with torch.no_grad(): # juste pour voir comment ca evolue du cote texte du coup
                # --- CLIP ---
                
                inputs = processor(images=images, text=captions, return_tensors="pt", padding=True) #.to(device)

                outputs = clip_model(**inputs)
                img_emb = outputs.image_embeds
                txt_emb = outputs.text_embeds

                loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)

            
            optimizer.zero_grad()
            loss_simclr.backward()
            optimizer.step()

            losses_simclr.append(loss_simclr.item())
            losses_clip.append(loss_clip.item())

        # --- Validation ---
        sim_val, clip_val, sum_val = compute_validation_losses(val_loader, clip_model, proj, transform)

        print(f"[Epoch {epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
        losses_simclr_all_epoch.append(np.mean(losses_simclr))
        losses_clip_all_epoch.append(np.mean(losses_clip))
        losses_simclr_val_all_epoch.append(sim_val)
        losses_clip_val_all_epoch.append(clip_val)
        losses_sum_val_all_epoch.append(sum_val)

        
        torch.save({
            "clip": clip_model.state_dict(),
            "proj": proj.state_dict()
        }, weightPath+f"SLIP_simclr_then_clip_then_both_{name}/{epoch}.pth")
    


    # --- CLIP PART SECOND ---

    for epoch in range(int(epochs/3), int(2*epochs/3)):
        losses_simclr = []
        losses_clip = []

        for (img, img_paths, captions, idx) in tqdm(train_loader):
            images = [Image.open(p).convert("RGB") for p in img_paths]
            
            with torch.no_grad():
                # --- SIMCLR  ---
                
                x = img #.to(device)
                x2 = transform(img) #.to(device)

                v1 = clip_model.get_image_features(pixel_values=x)
                v2 = clip_model.get_image_features(pixel_values=x2)

                z1 = proj(v1)
                z2 = proj(v2)

                loss_simclr = infoNCEloss(z1, z2, temperature)

            # --- CLIP ---
            
            inputs = processor(images=images, text=captions, return_tensors="pt", padding=True) #.to(device)

            outputs = clip_model(**inputs)
            
           
            img_emb = outputs.image_embeds

            txt_emb = outputs.text_embeds

            loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)
            
            optimizer.zero_grad()
            loss_clip.backward()
            optimizer.step()

            losses_simclr.append(loss_simclr.item())
            losses_clip.append(loss_clip.item())

        # --- Validation ---
        sim_val, clip_val, sum_val = compute_validation_losses(val_loader, clip_model, proj, transform)

        print(f"[Epoch {epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
        losses_simclr_all_epoch.append(np.mean(losses_simclr))
        losses_clip_all_epoch.append(np.mean(losses_clip))
        losses_simclr_val_all_epoch.append(sim_val)
        losses_clip_val_all_epoch.append(clip_val)
        losses_sum_val_all_epoch.append(sum_val)

        torch.save({
            "clip": clip_model.state_dict(),
            "proj": proj.state_dict()
        }, weightPath+f"SLIP_simclr_then_clip_then_both_{name}/{epoch}.pth")

    

    # JOINT PART
    for epoch in range(int(2*epochs/3),epochs):
        losses_simclr = []
        losses_clip = []

        for (img, img_paths, captions, idx) in tqdm(train_loader):
            images = [Image.open(p).convert("RGB") for p in img_paths]
            
            # --- SIMCLR PART ---
            
            x = img #.to(device)
            x2 = transform(img) #.to(device)

            v1 = clip_model.get_image_features(pixel_values=x)
            v2 = clip_model.get_image_features(pixel_values=x2)

            z1 = proj(v1)
            z2 = proj(v2)

            loss_simclr = infoNCEloss(z1, z2, temperature)

            # --- CLIP PART ---
            
            inputs = processor(images=images, text=captions, return_tensors="pt", padding=True) #.to(device)

            outputs = clip_model(**inputs)
            img_emb = outputs.image_embeds
            txt_emb = outputs.text_embeds

            loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)

            # --- TOTAL LOSS ---
            loss_total = lambda_simclr * loss_simclr + lambda_clip * loss_clip
            
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            losses_simclr.append(loss_simclr.item())
            losses_clip.append(loss_clip.item())

        # --- Validation ---
        sim_val, clip_val, sum_val = compute_validation_losses(val_loader, clip_model, proj, transform)

        print(f"[Epoch {epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
        losses_simclr_all_epoch.append(np.mean(losses_simclr))
        losses_clip_all_epoch.append(np.mean(losses_clip))
        losses_simclr_val_all_epoch.append(sim_val)
        losses_clip_val_all_epoch.append(clip_val)
        losses_sum_val_all_epoch.append(sum_val)

        torch.save({
            "clip": clip_model.state_dict(),
            "proj": proj.state_dict()
        }, weightPath+f"SLIP_simclr_then_clip_then_both_{name}/{epoch}.pth")
    

    plt.plot(losses_simclr_all_epoch, label="InfoNCE Loss", color='b')
    plt.plot(losses_clip_all_epoch, label="CLIP Loss", color='g')
    plt.plot(losses_simclr_val_all_epoch, label="Val InfoNCE", color='r')
    plt.plot(losses_clip_val_all_epoch,   label="Val CLIP", color='orange')
    plt.plot(losses_sum_val_all_epoch,   label="Val CLIP+InfoNCE", color='yellow')
    plt.axvline(x=int(epochs/3), color='r', linestyle='--')
    plt.axvline(x=int(2*epochs/3), color='r', linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("SLIP Training: SimCLR then CLIP then both")
    plt.legend()
    plt.savefig(lossPath+f"loss_slip_simclr_then_clip_then_both_{name}.png")
    plt.close()



def train_simclr_then_clip_then_both_new(train_loader, val_loader, transform, name, epochs=30, lambda_clip = 1.0, lambda_simclr = 1.0):
    model.train()
    losses_simclr_all_epoch = []
    losses_clip_all_epoch = []
    losses_simclr_val_all_epoch = []
    losses_clip_val_all_epoch = []
    losses_sum_val_all_epoch = []
    
    if not os.path.exists(weightPath+f"SLIP_simclr_then_clip_then_both_{name}"):
        os.mkdir(weightPath+f"SLIP_simclr_then_clip_then_both_{name}")


    # --- SIMCLR PART FIRST ---

    for epoch in range(int(epochs/3)):
        losses_simclr = []
        losses_clip = []

        for (img, img_paths, captions, idx) in tqdm(train_loader):
            
            # --- SIMCLR ---
            
            x = img #.to(device)
            x2 = transform(img) #.to(device)
            
            v1 = model.encode_image(x)
            v2 = model.encode_image(x2)

            loss_simclr = infoNCEloss(v1, v2, temperature)

            with torch.no_grad(): # juste pour voir comment ca evolue du cote texte du coup
                # --- CLIP ---
                
                text = clip.tokenize(list(captions))#.to(device)
                img_emb, txt_emb = model(img, text)

                loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)

            
            optimizer.zero_grad()
            loss_simclr.backward()
            optimizer.step()

            losses_simclr.append(loss_simclr.item())
            losses_clip.append(loss_clip.item())

        # --- Validation ---
        sim_val, clip_val, sum_val = compute_validation_losses_new(val_loader, model, transform)

        print(f"[Epoch {epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
        losses_simclr_all_epoch.append(np.mean(losses_simclr))
        losses_clip_all_epoch.append(np.mean(losses_clip))
        losses_simclr_val_all_epoch.append(sim_val)
        losses_clip_val_all_epoch.append(clip_val)
        losses_sum_val_all_epoch.append(sum_val)

        
        torch.save({
            "clip": model.state_dict(),
        }, weightPath+f"SLIP_simclr_then_clip_then_both_{name}/{epoch}.pth")
    


    # --- CLIP PART SECOND ---

    for epoch in range(int(epochs/3), int(2*epochs/3)):
        losses_simclr = []
        losses_clip = []

        for (img, img_paths, captions, idx) in tqdm(train_loader):
            
            with torch.no_grad():
                # --- SIMCLR  ---
                
                x = img #.to(device)
                x2 = transform(img) #.to(device)

                v1 = model.encode_image(x)
                v2 = model.encode_image(x2)

                loss_simclr = infoNCEloss(v1, v2, temperature)

            # --- CLIP ---
            
            text = clip.tokenize(list(captions))#.to(device)
            img_emb, txt_emb = model(img, text)

            loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)
            
            optimizer.zero_grad()
            loss_clip.backward()
            optimizer.step()

            losses_simclr.append(loss_simclr.item())
            losses_clip.append(loss_clip.item())

        # --- Validation ---
        sim_val, clip_val, sum_val = compute_validation_losses_new(val_loader, model, transform)

        print(f"[Epoch {epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
        losses_simclr_all_epoch.append(np.mean(losses_simclr))
        losses_clip_all_epoch.append(np.mean(losses_clip))
        losses_simclr_val_all_epoch.append(sim_val)
        losses_clip_val_all_epoch.append(clip_val)
        losses_sum_val_all_epoch.append(sum_val)

        torch.save({
            "clip": model.state_dict(),
        }, weightPath+f"SLIP_simclr_then_clip_then_both_{name}/{epoch}.pth")

    

    # JOINT PART
    for epoch in range(int(2*epochs/3),epochs):
        losses_simclr = []
        losses_clip = []

        for (img, img_paths, captions, idx) in tqdm(train_loader):
            
            # --- SIMCLR PART ---
            
            x = img #.to(device)
            x2 = transform(img) #.to(device)

            v1 = model.encode_image(x)
            v2 = model.encode_image(x2)

            loss_simclr = infoNCEloss(v1, v2, temperature)

            # --- CLIP PART ---
            
            text = clip.tokenize(list(captions))#.to(device)
            img_emb, txt_emb = model(img, text)

            loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)

            # --- TOTAL LOSS ---
            loss_total = lambda_simclr * loss_simclr + lambda_clip * loss_clip
            
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            losses_simclr.append(loss_simclr.item())
            losses_clip.append(loss_clip.item())

        # --- Validation ---
        sim_val, clip_val, sum_val = compute_validation_losses_new(val_loader, model, transform)

        print(f"[Epoch {epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
        losses_simclr_all_epoch.append(np.mean(losses_simclr))
        losses_clip_all_epoch.append(np.mean(losses_clip))
        losses_simclr_val_all_epoch.append(sim_val)
        losses_clip_val_all_epoch.append(clip_val)
        losses_sum_val_all_epoch.append(sum_val)

        torch.save({
            "clip": model.state_dict(),
        }, weightPath+f"SLIP_simclr_then_clip_then_both_{name}/{epoch}.pth")
    

    plt.plot(losses_simclr_all_epoch, label="InfoNCE Loss", color='b')
    plt.plot(losses_clip_all_epoch, label="CLIP Loss", color='g')
    plt.plot(losses_simclr_val_all_epoch, label="Val InfoNCE", color='r')
    plt.plot(losses_clip_val_all_epoch,   label="Val CLIP", color='orange')
    plt.plot(losses_sum_val_all_epoch,   label="Val CLIP+InfoNCE", color='yellow')
    plt.axvline(x=int(epochs/3), color='r', linestyle='--')
    plt.axvline(x=int(2*epochs/3), color='r', linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("SLIP Training: SimCLR then CLIP then both")
    plt.legend()
    plt.savefig(lossPath+f"loss_slip_simclr_then_clip_then_both_{name}.png")
    plt.close()



def train_alternate_then_clip(train_loader, val_loader, transform, name, epochs_total=60, step_alternate=5, epochs_clip_only=15):
    clip_model.train()
    proj.train()
    losses_simclr_all_epoch = []
    losses_clip_all_epoch = []
    losses_simclr_val_all_epoch = []
    losses_clip_val_all_epoch = []
    losses_sum_val_all_epoch = []
    
    epochs = epochs_total - epochs_clip_only
    if not os.path.exists(weightPath+f"SLIP_alternate_then_clip_{name}"):
        os.mkdir(weightPath+f"SLIP_alternate_then_clip_{name}")


    for epoch in range(int(epochs/(2*step_alternate))):
        

        for i in range(step_alternate):
            losses_simclr = []
            losses_clip = []

            for (img, img_paths, captions, idx) in tqdm(train_loader):
                images = [Image.open(p).convert("RGB") for p in img_paths]


                with torch.no_grad():
                
                    # --- SIMCLR ---
                    
                    x = img #.to(device)
                    x2 = transform(img) #.to(device)
                    
                    v1 = clip_model.get_image_features(pixel_values=x)
                    v2 = clip_model.get_image_features(pixel_values=x2)

                    z1 = proj(v1)
                    z2 = proj(v2)

                    loss_simclr = infoNCEloss(z1, z2, temperature)

                # --- CLIP ---
                    
                inputs = processor(images=images, text=captions, return_tensors="pt", padding=True) #.to(device)

                outputs = clip_model(**inputs) 
                img_emb = outputs.image_embeds
                txt_emb = outputs.text_embeds

                loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)

                
                optimizer.zero_grad()
                loss_clip.backward()
                optimizer.step()

                losses_simclr.append(loss_simclr.item())
                losses_clip.append(loss_clip.item())
            
            # --- Validation ---
            sim_val, clip_val, sum_val = compute_validation_losses(val_loader, clip_model, proj, transform)

            print(f"[Epoch {i+step_alternate*2*epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
            losses_simclr_all_epoch.append(np.mean(losses_simclr))
            losses_clip_all_epoch.append(np.mean(losses_clip))
            losses_simclr_val_all_epoch.append(sim_val)
            losses_clip_val_all_epoch.append(clip_val)
            losses_sum_val_all_epoch.append(sum_val)

            torch.save({
                "clip": clip_model.state_dict(),
                "proj": proj.state_dict()
            }, weightPath+f"SLIP_alternate_then_clip_{name}/{i+step_alternate*2*epoch}.pth")
    


        # --- SIMCLR PART ---

        for i in range(step_alternate):
            losses_simclr = []
            losses_clip = []
    

            for (img, img_paths, captions, idx) in tqdm(train_loader):
                images = [Image.open(p).convert("RGB") for p in img_paths]
                
                
                # --- SIMCLR  ---
                    
                x = img #.to(device)
                x2 = transform(img) #.to(device)

                v1 = clip_model.get_image_features(pixel_values=x)
                v2 = clip_model.get_image_features(pixel_values=x2)

                z1 = proj(v1)
                z2 = proj(v2)

                loss_simclr = infoNCEloss(z1, z2, temperature)


                with torch.no_grad():
                    # --- CLIP ---
                    
                    inputs = processor(images=images, text=captions, return_tensors="pt", padding=True) #.to(device)

                    outputs = clip_model(**inputs)
                    img_emb = outputs.image_embeds
                    txt_emb = outputs.text_embeds

                    loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)
                
                optimizer.zero_grad()
                loss_simclr.backward()
                optimizer.step()

                losses_simclr.append(loss_simclr.item())
                losses_clip.append(loss_clip.item())

            # --- Validation ---
            sim_val, clip_val, sum_val = compute_validation_losses(val_loader, clip_model, proj, transform)

            print(f"[Epoch {i+step_alternate*2*epoch+step_alternate}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
            losses_simclr_all_epoch.append(np.mean(losses_simclr))
            losses_clip_all_epoch.append(np.mean(losses_clip))
            losses_simclr_val_all_epoch.append(sim_val)
            losses_clip_val_all_epoch.append(clip_val)
            losses_sum_val_all_epoch.append(sum_val)

            torch.save({
                "clip": clip_model.state_dict(),
                "proj": proj.state_dict()
            }, weightPath+f"SLIP_alternate_then_clip_{name}/{i+step_alternate*2*epoch+step_alternate}.pth")
        

    for epoch in range(epochs, epochs_total):
        losses_simclr = []
        losses_clip = []

        for (img, img_paths, captions, idx) in tqdm(train_loader):
            images = [Image.open(p).convert("RGB") for p in img_paths]
            
            
            with torch.no_grad():
                # --- SIMCLR  ---
                    
                x = img #.to(device)
                x2 = transform(img) #.to(device)

                v1 = clip_model.get_image_features(pixel_values=x)
                v2 = clip_model.get_image_features(pixel_values=x2)

                z1 = proj(v1)
                z2 = proj(v2)

                loss_simclr = infoNCEloss(z1, z2, temperature)

            # --- CLIP ---
            
            inputs = processor(images=images, text=captions, return_tensors="pt", padding=True) #.to(device)

            outputs = clip_model(**inputs)
            
           
            img_emb = outputs.image_embeds

            txt_emb = outputs.text_embeds

            loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)
            
            optimizer.zero_grad()
            loss_clip.backward()
            optimizer.step()

            losses_simclr.append(loss_simclr.item())
            losses_clip.append(loss_clip.item())

        # --- Validation ---
        sim_val, clip_val, sum_val = compute_validation_losses(val_loader, clip_model, proj, transform)
        print(f"[Epoch {epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
        losses_simclr_all_epoch.append(np.mean(losses_simclr))
        losses_clip_all_epoch.append(np.mean(losses_clip))
        losses_simclr_val_all_epoch.append(sim_val)
        losses_clip_val_all_epoch.append(clip_val)
        losses_sum_val_all_epoch.append(sum_val)

        torch.save({
            "clip": clip_model.state_dict(),
            "proj": proj.state_dict()
        }, weightPath+f"SLIP_alternate_then_clip_{name}/{epoch}.pth")

    plt.plot(losses_simclr_all_epoch, label="InfoNCE Loss", color='b')
    plt.plot(losses_clip_all_epoch, label="CLIP Loss", color='g')
    plt.plot(losses_simclr_val_all_epoch, label="Val InfoNCE", color='r')
    plt.plot(losses_clip_val_all_epoch,   label="Val CLIP", color='orange')
    plt.plot(losses_sum_val_all_epoch,   label="Val CLIP+InfoNCE", color='yellow')
    plt.axvline(x=epochs, color='r', linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("SLIP Training: alternate then only CLIP")
    plt.legend()
    plt.savefig(lossPath+f"loss_slip_alternate_then_clip_{name}.png")
    plt.close()






def train_alternate_then_clip_new(train_loader, val_loader, transform, name, epochs_total=60, step_alternate=5, epochs_clip_only=15):
    model.train()
    losses_simclr_all_epoch = []
    losses_clip_all_epoch = []
    losses_simclr_val_all_epoch = []
    losses_clip_val_all_epoch = []
    losses_sum_val_all_epoch = []
    
    epochs = epochs_total - epochs_clip_only
    if not os.path.exists(weightPath+f"SLIP_alternate_then_clip_{name}"):
        os.mkdir(weightPath+f"SLIP_alternate_then_clip_{name}")


    for epoch in range(int(epochs/(2*step_alternate))):
        

        for i in range(step_alternate):
            losses_simclr = []
            losses_clip = []

            for (img, img_paths, captions, idx) in tqdm(train_loader):

                with torch.no_grad():
                
                    # --- SIMCLR ---
                    
                    x = img #.to(device)
                    x2 = transform(img) #.to(device)
                    
                    v1 = model.encode_image(x)
                    v2 = model.encode_image(x2)

                    loss_simclr = infoNCEloss(v1, v2, temperature)


                # --- CLIP ---
                    
                text = clip.tokenize(list(captions))#.to(device)
                img_emb, txt_emb = model(img, text)

                loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)

                
                optimizer.zero_grad()
                loss_clip.backward()
                optimizer.step()

                losses_simclr.append(loss_simclr.item())
                losses_clip.append(loss_clip.item())
            
            # --- Validation ---
            sim_val, clip_val, sum_val = compute_validation_losses_new(val_loader, model, transform)

            print(f"[Epoch {i+step_alternate*2*epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
            losses_simclr_all_epoch.append(np.mean(losses_simclr))
            losses_clip_all_epoch.append(np.mean(losses_clip))
            losses_simclr_val_all_epoch.append(sim_val)
            losses_clip_val_all_epoch.append(clip_val)
            losses_sum_val_all_epoch.append(sum_val)

            torch.save({
                "clip": model.state_dict(),
            }, weightPath+f"SLIP_alternate_then_clip_{name}/{i+step_alternate*2*epoch}.pth")
    


        # --- SIMCLR PART ---

        for i in range(step_alternate):
            losses_simclr = []
            losses_clip = []
    

            for (img, img_paths, captions, idx) in tqdm(train_loader):
                
                # --- SIMCLR  ---
                    
                x = img #.to(device)
                x2 = transform(img) #.to(device)

                v1 = model.encode_image(x)
                v2 = model.encode_image(x2)

                loss_simclr = infoNCEloss(v1, v2, temperature)


                with torch.no_grad():
                    # --- CLIP ---
                    
                    text = clip.tokenize(list(captions))#.to(device)
                    img_emb, txt_emb = model(img, text)

                    loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)
                
                optimizer.zero_grad()
                loss_simclr.backward()
                optimizer.step()

                losses_simclr.append(loss_simclr.item())
                losses_clip.append(loss_clip.item())

            # --- Validation ---
            sim_val, clip_val, sum_val = compute_validation_losses_new(val_loader, model, transform)

            print(f"[Epoch {i+step_alternate*2*epoch+step_alternate}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
            losses_simclr_all_epoch.append(np.mean(losses_simclr))
            losses_clip_all_epoch.append(np.mean(losses_clip))
            losses_simclr_val_all_epoch.append(sim_val)
            losses_clip_val_all_epoch.append(clip_val)
            losses_sum_val_all_epoch.append(sum_val)

            torch.save({
                "clip": model.state_dict(),
            }, weightPath+f"SLIP_alternate_then_clip_{name}/{i+step_alternate*2*epoch+step_alternate}.pth")
        

    for epoch in range(epochs, epochs_total):
        losses_simclr = []
        losses_clip = []

        for (img, img_paths, captions, idx) in tqdm(train_loader):
            
            with torch.no_grad():
                # --- SIMCLR  ---
                    
                x = img #.to(device)
                x2 = transform(img) #.to(device)

                v1 = model.encode_image(x)
                v2 = model.encode_image(x2)

                loss_simclr = infoNCEloss(v1, v2, temperature)

            # --- CLIP ---
            
            text = clip.tokenize(list(captions))#.to(device)
            img_emb, txt_emb = model(img, text)

            loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)
            
            optimizer.zero_grad()
            loss_clip.backward()
            optimizer.step()

            losses_simclr.append(loss_simclr.item())
            losses_clip.append(loss_clip.item())

        # --- Validation ---
        sim_val, clip_val, sum_val = compute_validation_losses_new(val_loader, model, transform)
        print(f"[Epoch {epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
        losses_simclr_all_epoch.append(np.mean(losses_simclr))
        losses_clip_all_epoch.append(np.mean(losses_clip))
        losses_simclr_val_all_epoch.append(sim_val)
        losses_clip_val_all_epoch.append(clip_val)
        losses_sum_val_all_epoch.append(sum_val)

        torch.save({
            "clip": model.state_dict(),
        }, weightPath+f"SLIP_alternate_then_clip_{name}/{epoch}.pth")

    plt.plot(losses_simclr_all_epoch, label="InfoNCE Loss", color='b')
    plt.plot(losses_clip_all_epoch, label="CLIP Loss", color='g')
    plt.plot(losses_simclr_val_all_epoch, label="Val InfoNCE", color='r')
    plt.plot(losses_clip_val_all_epoch,   label="Val CLIP", color='orange')
    plt.plot(losses_sum_val_all_epoch,   label="Val CLIP+InfoNCE", color='yellow')
    plt.axvline(x=epochs, color='r', linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("SLIP Training: alternate then only CLIP")
    plt.legend()
    plt.savefig(lossPath+f"loss_slip_alternate_then_clip_{name}.png")
    plt.close()




def train_alternate_then_simclr(train_loader, val_loader, transform, name, epochs_total=60, step_alternate=5, epochs_clip_only=15):
    clip_model.train()
    proj.train()
    losses_simclr_all_epoch = []
    losses_clip_all_epoch = []
    losses_simclr_val_all_epoch = []
    losses_clip_val_all_epoch = []
    losses_sum_val_all_epoch = []
    
    epochs = epochs_total - epochs_clip_only
    if not os.path.exists(weightPath+f"SLIP_alternate_then_simclr_{name}"):
        os.mkdir(weightPath+f"SLIP_alternate_then_simclr_{name}")


    for epoch in range(int(epochs/(2*step_alternate))):
        

        for i in range(step_alternate):
            losses_simclr = []
            losses_clip = []

            for (img, img_paths, captions, idx) in tqdm(train_loader):
                images = [Image.open(p).convert("RGB") for p in img_paths]


                with torch.no_grad():
                
                    # --- SIMCLR ---
                    
                    x = img #.to(device)
                    x2 = transform(img) #.to(device)
                    
                    v1 = clip_model.get_image_features(pixel_values=x)
                    v2 = clip_model.get_image_features(pixel_values=x2)

                    z1 = proj(v1)
                    z2 = proj(v2)

                    loss_simclr = infoNCEloss(z1, z2, temperature)

                # --- CLIP ---
                    
                inputs = processor(images=images, text=captions, return_tensors="pt", padding=True) #.to(device)

                outputs = clip_model(**inputs) 
                img_emb = outputs.image_embeds
                txt_emb = outputs.text_embeds

                loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)

                
                optimizer.zero_grad()
                loss_clip.backward()
                optimizer.step()

                losses_simclr.append(loss_simclr.item())
                losses_clip.append(loss_clip.item())
            
            # --- Validation ---
            sim_val, clip_val, sum_val = compute_validation_losses(val_loader, clip_model, proj, transform)

            print(f"[Epoch {i+step_alternate*2*epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
            losses_simclr_all_epoch.append(np.mean(losses_simclr))
            losses_clip_all_epoch.append(np.mean(losses_clip))
            losses_simclr_val_all_epoch.append(sim_val)
            losses_clip_val_all_epoch.append(clip_val)
            losses_sum_val_all_epoch.append(sum_val)

            torch.save({
                "clip": clip_model.state_dict(),
                "proj": proj.state_dict()
            }, weightPath+f"SLIP_alternate_then_simclr_{name}/{i+step_alternate*2*epoch}.pth")
    


        # --- SIMCLR PART ---

        for i in range(step_alternate):
            losses_simclr = []
            losses_clip = []
    

            for (img, img_paths, captions, idx) in tqdm(train_loader):
                images = [Image.open(p).convert("RGB") for p in img_paths]
                
                
                # --- SIMCLR  ---
                    
                x = img #.to(device)
                x2 = transform(img) #.to(device)

                v1 = clip_model.get_image_features(pixel_values=x)
                v2 = clip_model.get_image_features(pixel_values=x2)

                z1 = proj(v1)
                z2 = proj(v2)

                loss_simclr = infoNCEloss(z1, z2, temperature)


                with torch.no_grad():
                    # --- CLIP ---
                    
                    inputs = processor(images=images, text=captions, return_tensors="pt", padding=True) #.to(device)

                    outputs = clip_model(**inputs)
                    img_emb = outputs.image_embeds
                    txt_emb = outputs.text_embeds

                    loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)
                
                optimizer.zero_grad()
                loss_simclr.backward()
                optimizer.step()

                losses_simclr.append(loss_simclr.item())
                losses_clip.append(loss_clip.item())

            # --- Validation ---
            sim_val, clip_val, sum_val = compute_validation_losses(val_loader, clip_model, proj, transform)

            print(f"[Epoch {i+step_alternate*2*epoch+step_alternate}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
            losses_simclr_all_epoch.append(np.mean(losses_simclr))
            losses_clip_all_epoch.append(np.mean(losses_clip))
            losses_simclr_val_all_epoch.append(sim_val)
            losses_clip_val_all_epoch.append(clip_val)
            losses_sum_val_all_epoch.append(sum_val)

            torch.save({
                "clip": clip_model.state_dict(),
                "proj": proj.state_dict()
            }, weightPath+f"SLIP_alternate_then_simclr_{name}/{i+step_alternate*2*epoch+step_alternate}.pth")
        

    for epoch in range(epochs, epochs_total):
        losses_simclr = []
        losses_clip = []

        for (img, img_paths, captions, idx) in tqdm(train_loader):
            images = [Image.open(p).convert("RGB") for p in img_paths]
            
            # --- SIMCLR  ---
                    
            x = img #.to(device)
            x2 = transform(img) #.to(device)

            v1 = clip_model.get_image_features(pixel_values=x)
            v2 = clip_model.get_image_features(pixel_values=x2)

            z1 = proj(v1)
            z2 = proj(v2)

            loss_simclr = infoNCEloss(z1, z2, temperature)
            
            with torch.no_grad():
                # --- CLIP ---
                
                inputs = processor(images=images, text=captions, return_tensors="pt", padding=True) #.to(device)

                outputs = clip_model(**inputs)
                
            
                img_emb = outputs.image_embeds

                txt_emb = outputs.text_embeds

                loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)
            
            optimizer.zero_grad()
            loss_simclr.backward()
            optimizer.step()

            losses_simclr.append(loss_simclr.item())
            losses_clip.append(loss_clip.item())

        # --- Validation ---
        sim_val, clip_val, sum_val = compute_validation_losses(val_loader, clip_model, proj, transform)
        print(f"[Epoch {epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
        losses_simclr_all_epoch.append(np.mean(losses_simclr))
        losses_clip_all_epoch.append(np.mean(losses_clip))
        losses_simclr_val_all_epoch.append(sim_val)
        losses_clip_val_all_epoch.append(clip_val)
        losses_sum_val_all_epoch.append(sum_val)

        torch.save({
            "clip": clip_model.state_dict(),
            "proj": proj.state_dict()
        }, weightPath+f"SLIP_alternate_then_simclr_{name}/{epoch}.pth")

    plt.plot(losses_simclr_all_epoch, label="InfoNCE Loss", color='b')
    plt.plot(losses_clip_all_epoch, label="CLIP Loss", color='g')
    plt.plot(losses_simclr_val_all_epoch, label="Val InfoNCE", color='r')
    plt.plot(losses_clip_val_all_epoch,   label="Val CLIP", color='orange')
    plt.plot(losses_sum_val_all_epoch,   label="Val CLIP+InfoNCE", color='yellow')
    plt.axvline(x=epochs, color='r', linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("SLIP Training: alternate then only InfoNCE")
    plt.legend()
    plt.savefig(lossPath+f"loss_slip_alternate_then_simclr_{name}.png")
    plt.close()




def train_alternate_then_simclr_new(train_loader, val_loader, transform, name, epochs_total=60, step_alternate=5, epochs_clip_only=15):
    model.train()
    losses_simclr_all_epoch = []
    losses_clip_all_epoch = []
    losses_simclr_val_all_epoch = []
    losses_clip_val_all_epoch = []
    losses_sum_val_all_epoch = []
    
    epochs = epochs_total - epochs_clip_only
    if not os.path.exists(weightPath+f"SLIP_alternate_then_simclr_{name}"):
        os.mkdir(weightPath+f"SLIP_alternate_then_simclr_{name}")


    for epoch in range(int(epochs/(2*step_alternate))):
        
        for i in range(step_alternate):
            losses_simclr = []
            losses_clip = []

            for (img, img_paths, captions, idx) in tqdm(train_loader):

                with torch.no_grad():
                
                    # --- SIMCLR ---
                    
                    x = img #.to(device)
                    x2 = transform(img) #.to(device)
                
                    v1 = model.encode_image(x)
                    v2 = model.encode_image(x2)
                    loss_simclr = infoNCEloss(v1, v2, temperature)

                # --- CLIP ---
                    
                text = clip.tokenize(list(captions))#.to(device)
                img_emb, txt_emb = model(img, text)

                loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)

                
                optimizer.zero_grad()
                loss_clip.backward()
                optimizer.step()

                losses_simclr.append(loss_simclr.item())
                losses_clip.append(loss_clip.item())
            
            # --- Validation ---
            sim_val, clip_val, sum_val = compute_validation_losses_new(val_loader, model, transform)

            print(f"[Epoch {i+step_alternate*2*epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
            losses_simclr_all_epoch.append(np.mean(losses_simclr))
            losses_clip_all_epoch.append(np.mean(losses_clip))
            losses_simclr_val_all_epoch.append(sim_val)
            losses_clip_val_all_epoch.append(clip_val)
            losses_sum_val_all_epoch.append(sum_val)

            torch.save({
                "clip": model.state_dict(),
            }, weightPath+f"SLIP_alternate_then_simclr_{name}/{i+step_alternate*2*epoch}.pth")
    


        # --- SIMCLR PART ---

        for i in range(step_alternate):
            losses_simclr = []
            losses_clip = []
    

            for (img, img_paths, captions, idx) in tqdm(train_loader):
                
                # --- SIMCLR  ---
                    
                x = img #.to(device)
                x2 = transform(img) #.to(device)

                v1 = model.encode_image(x)
                v2 = model.encode_image(x2)

                loss_simclr = infoNCEloss(v1, v2, temperature)


                with torch.no_grad():
                    # --- CLIP ---
                    text = clip.tokenize(list(captions))#.to(device)
                    img_emb, txt_emb = model(img, text)

                    loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)
                
                optimizer.zero_grad()
                loss_simclr.backward()
                optimizer.step()

                losses_simclr.append(loss_simclr.item())
                losses_clip.append(loss_clip.item())

            # --- Validation ---
            sim_val, clip_val, sum_val = compute_validation_losses_new(val_loader, model, transform)

            print(f"[Epoch {i+step_alternate*2*epoch+step_alternate}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
            losses_simclr_all_epoch.append(np.mean(losses_simclr))
            losses_clip_all_epoch.append(np.mean(losses_clip))
            losses_simclr_val_all_epoch.append(sim_val)
            losses_clip_val_all_epoch.append(clip_val)
            losses_sum_val_all_epoch.append(sum_val)

            torch.save({
                "clip": model.state_dict(),
            }, weightPath+f"SLIP_alternate_then_simclr_{name}/{i+step_alternate*2*epoch+step_alternate}.pth")
        

    for epoch in range(epochs, epochs_total):
        losses_simclr = []
        losses_clip = []

        for (img, img_paths, captions, idx) in tqdm(train_loader):
            
            # --- SIMCLR  ---
                    
            x = img #.to(device)
            x2 = transform(img) #.to(device)

            v1 = model.encode_image(x)
            v2 = model.encode_image(x2)

            loss_simclr = infoNCEloss(v1, v2, temperature)
            
            with torch.no_grad():
                # --- CLIP ---
                
                text = clip.tokenize(list(captions))#.to(device)
                img_emb, txt_emb = model(img, text)

                loss_clip = clip_similarity_loss(img_emb, txt_emb, temperature)
            
            optimizer.zero_grad()
            loss_simclr.backward()
            optimizer.step()

            losses_simclr.append(loss_simclr.item())
            losses_clip.append(loss_clip.item())

        # --- Validation ---
        sim_val, clip_val, sum_val = compute_validation_losses_new(val_loader, model, transform)
        print(f"[Epoch {epoch}] SimCLR: {np.mean(losses_simclr):.4f} | CLIP: {np.mean(losses_clip):.4f}")
        losses_simclr_all_epoch.append(np.mean(losses_simclr))
        losses_clip_all_epoch.append(np.mean(losses_clip))
        losses_simclr_val_all_epoch.append(sim_val)
        losses_clip_val_all_epoch.append(clip_val)
        losses_sum_val_all_epoch.append(sum_val)

        torch.save({
            "clip": model.state_dict(),
        }, weightPath+f"SLIP_alternate_then_simclr_{name}/{epoch}.pth")

    plt.plot(losses_simclr_all_epoch, label="InfoNCE Loss", color='b')
    plt.plot(losses_clip_all_epoch, label="CLIP Loss", color='g')
    plt.plot(losses_simclr_val_all_epoch, label="Val InfoNCE", color='r')
    plt.plot(losses_clip_val_all_epoch,   label="Val CLIP", color='orange')
    plt.plot(losses_sum_val_all_epoch,   label="Val CLIP+InfoNCE", color='yellow')
    plt.axvline(x=epochs, color='r', linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("SLIP Training: alternate then only InfoNCE")
    plt.legend()
    plt.savefig(lossPath+f"loss_slip_alternate_then_simclr_{name}.png")
    plt.close()





#######################################################




try:
    train_alternate_clip_simclr_new(train_loader, val_loader, transform, "test1_new", epochs=60)
except Exception as e:
    print(f"Error during training alternate_clip_simclr_new:\n{e}")



model, _ = clip.load("ViT-B/32", device)
optimizer = optim.AdamW(list(model.parameters()), lr=5e-6)


try:
    train_joint_new(train_loader, val_loader, transform, "test1_new", epochs=20)
except Exception as e:
    print(f"Error during training train_joint_new:\n{e}")

model, _ = clip.load("ViT-B/32", device)
optimizer = optim.AdamW(list(model.parameters()), lr=5e-6)

try:
    train_simclr_then_clip_new(train_loader, val_loader, transform, "test1_new", epochs=30)
except Exception as e:
    print(f"Error during training train_simclr_then_clip_new:\n{e}")

model, _ = clip.load("ViT-B/32", device)
optimizer = optim.AdamW(list(model.parameters()), lr=5e-6)

try:
    train_simclr_then_clip_new(train_loader, val_loader, transform, "test1_new_freeze", epochs=30, freeze_simclr=True)
except Exception as e:
    print(f"Error during training train_simclr_then_clip_new_freeze:\n{e}")

model, _ = clip.load("ViT-B/32", device)
optimizer = optim.AdamW(list(model.parameters()), lr=5e-6)

try:
    train_clip_then_simclr_new(train_loader, val_loader, transform, "test1_new", epochs=30)
except Exception as e:
    print(f"Error during training train_clip_then_simclr_new:\n{e}")

model, _ = clip.load("ViT-B/32", device)
optimizer = optim.AdamW(list(model.parameters()), lr=5e-6)


try:
    train_simclr_then_clip_then_both_new(train_loader, val_loader, transform, "test1_new", epochs=30)
except Exception as e:
    print(f"Error during training train_simclr_then_clip_then_both_new:\n{e}")

model, _ = clip.load("ViT-B/32", device)
optimizer = optim.AdamW(list(model.parameters()), lr=5e-6)

try:
    train_joint_new(train_loader, val_loader, transform, "test2_new_05clip_1simclr", epochs=20, lambda_clip = 0.5, lambda_simclr = 1.0)
except Exception as e:
    print(f"Error during training train_joint_new_05clip_1simclr:\n{e}")

model, _ = clip.load("ViT-B/32", device)
optimizer = optim.AdamW(list(model.parameters()), lr=5e-6)

try:
    train_joint_new(train_loader, val_loader, transform, "test3_new_1clip_05simclr", epochs=20, lambda_clip = 1.0, lambda_simclr = 0.5)
except Exception as e:
    print(f"Error during training train_joint_new_1clip_05simclr:\n{e}")
