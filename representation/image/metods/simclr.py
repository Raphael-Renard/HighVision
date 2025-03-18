import sys
path = "../../.."
if path not in sys.path:
    sys.path.insert(0, path)


from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from data_retrieval import lipade_groundtruth
from data_retrieval.tools.data_loader import getDataLoader
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from data_retrieval import lipade_groundtruth
from clustering.clustering import getPredictionFromThreshold
import clustering.evaluators as evaluators
import csv
import optuna

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

optimizerFunc = optim.Adam
temperature = 0.07
learningRate = 3e-4
batch_size = 512
workers = 2
extract = 500
epochs = 60
corpus = "lipade_groundtruth"
resultsPath = "./results/distance/" + corpus + "/"
csv_filename = "results.csv"



# Data
x,_,y = lipade_groundtruth.getDataset(mode = 'unique', uniform=True)

images = []
for i in range(500):
    images.append(Image.open(x[i]).convert('RGB'))


x = np.array(x[:500])
y = np.array(y[:500])
images = np.array(images)
print(images.shape)


trainLoader = getDataLoader(images, None, None, False, batch_size, True, num_workers=2)


# Test
xSim,_,ySim = lipade_groundtruth.getDataset(mode = 'similar', uniform=True)
_,_,y_test = lipade_groundtruth.getDataset(mode="similar")

imagesSim = []
for i in range(len(xSim)):
    imagesSim.append(Image.open(xSim[i]).convert('RGB'))

testLoader = getDataLoader(imagesSim, None, None, False, batch_size, shuffle=False, num_workers=2)



# Transformations

from degradations.methods.halftoning.floyd_steinberg import transforms_floyd_steinberg_halftoning
from degradations.methods.halftoning.atkinson import transforms_atkinson_dithering
from degradations.methods.halftoning.bayers_threshold import transforms_bayer_halftoning
from degradations.methods.halftoning.dot_traditional import transforms_dot_halftoning  # Import your halftoning methods
from degradations.methods.noise.gaussian_noise import transforms_add_gaussian_noise
from degradations.methods.noise.salt_and_pepper import transforms_add_salt_and_pepper_noise
from degradations.methods.noise.dirty_rollers import transforms_dirty_rollers
#from degradations.methods.noise.film_grain import transforms_apply_film_grain # Import your noise methods
from degradations.methods.paper.ink_bleed import transforms_ink_bleed  
from degradations.methods.paper.crumpled_paper import transforms_crumpled_paper
from degradations.methods.paper.folded_paper import transforms_folded_paper
from degradations.methods.paper.bleedthrough import transforms_bleedthrough
from degradations.methods.paper.scribbles import transforms_scribbles
from degradations.methods.paper.torn_paper import transforms_torn_paper
from degradations.methods.paper.stains import transforms_stains # Import your paper feel methods
from degradations.methods.human_corrections.erased_element import transforms_erased_element # Import your human correction methods
from degradations.methods.layout.picture_overlay import transforms_picture_overlay
from degradations.methods.layout.text_overlay import transforms_text_overlay # Import your layout methods


# Sepia
class transforms_SepiaFilter(nn.Module):
    def __init__(self):
        super(transforms_SepiaFilter, self).__init__()

    def __call__(self, batch):
        sepia_filter = torch.tensor([[0.393, 0.769, 0.189],
                                     [0.349, 0.686, 0.168],
                                     [0.272, 0.534, 0.131]], device=batch.device)
        batch = torch.einsum('ijkl,mj->imkl', batch, sepia_filter)
        return batch.clamp(0, 1)





# Representation

class SimCLR_Representation(nn.Module):
    def __init__(self, encoder, in_dim=2048, out_dim=128):
        super(SimCLR_Representation, self).__init__()
        self.encoder = encoder
        self.projection = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )
    
    def forward(self, x_):
        h = self.encoder(x_)
        z = self.projection(h)
        return z



# Loss
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


# Train
def train_simclr(transform, name):
    # ResNet50 - Stage 4
    representationEncoder = resnet18(weights=ResNet18_Weights.DEFAULT)
    representationEncoder.fc = nn.Identity()
    model = SimCLR_Representation(representationEncoder, in_dim=512).to(device)
    optimizer = optimizerFunc(model.parameters(), lr=learningRate)

    lastBatch = images.shape[0] // batch_size

    writer = SummaryWriter(log_dir=f"logs/SimCLR_{name}")

    model.train()
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    losses_all = []
    
    for epoch in range(epochs):
        losses = []
        for i,sampledMinibatch in enumerate(tqdm(trainLoader, desc="Epoch " + str(epoch))):
            x = sampledMinibatch.to(device)
            # Transformation
            x_2 = transform(x)
            # Representation
            z1 = model(x)
            z2 = model(x_2)
            # Loss
            loss = infoNCEloss(z1, z2, temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i != lastBatch:
                losses.append(loss.item())
        writer.add_scalar("Loss/train", torch.tensor(losses).mean(), epoch)
        losses_all.append(torch.tensor(losses).mean())
        torch.save(model.state_dict(), f"model_{name}.pth")
        scheduler.step()
    plt.plot(losses_all)
    plt.savefig(f"loss_{name}.png")
    plt.close()


    # Test 
    representations = []
    with torch.no_grad():
        for batch in testLoader:
            batch = model(batch.to(device))
            for repr in batch.tolist():
                representations.append(repr)

    sim = cosine_similarity(representations, representations)

    distance = 1 - (sim+1)/2
    distance -= np.diag(distance)

    np.save(resultsPath + f"simclr_{name}.npy", distance)

    method = f"simclr_{name}"

    thresholds_precision = 1000
    thresholds = np.linspace(0, 1, thresholds_precision)

    precisions, recalls, f1s = evaluators.p_r_f1_byThresholds(thresholds, distance, y)

    AP, bestThresholdIndex = evaluators.pr_curve(precisions, recalls, f1s, other=("Fei mAP", evaluators.fei_mAP(y, distance)), save="evaluation/" + corpus + "/" + method + ".png")

    thresholdsClass = np.linspace(0, 1, int(thresholds_precision / 10))
    precisions_per_class, recalls_per_class = evaluators.p_r_class_byThresholds(thresholdsClass, distance, y)
    fei = evaluators.fei_mAP(y, distance)
    sam = evaluators.goncalves_mAP(precisions_per_class, recalls_per_class)

    
    # Écriture des résultats
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["trial_name", "ap", "fei", "sam"])
        writer.writerow([name, AP, fei, sam])


    return (AP + fei + sam)/3





def objective(trial):
    # Sampling probabilities
    prob_crop = trial.suggest_float("prob_crop", 0.5, 1.0)
    prob_halftone = trial.suggest_float("prob_halftone", 0.0, 1.0)
    prob_layout = trial.suggest_float("prob_layout", 0.0, 1.0)
    prob_erased = trial.suggest_float("prob_erased", 0.0, 1.0)
    prob_noise = trial.suggest_float("prob_noise", 0.2, 1.0)
    prob_stains = trial.suggest_float("prob_stains", 0.2, 1.0)
    prob_texture = trial.suggest_float("prob_texture", 0.0, 1.0)
    prob_flip = trial.suggest_float("prob_flip", 0.0, 1.0)
    prob_blur = trial.suggest_float("prob_blur", 0.0, 1.0)
    prob_sepia = trial.suggest_float("prob_sepia", 0.0, 1.0)

    # Construire la transformation
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=images.shape[2], scale=(1/2, 1), ratio=(1, 1)) if trial.suggest_categorical("use_crop", [True, False]) else None,
        transforms.RandomApply([transforms.RandomChoice([
            transforms_floyd_steinberg_halftoning(),
            transforms_atkinson_dithering(),
            transforms_bayer_halftoning()
        ])], p=prob_halftone),
        transforms.RandomApply([transforms.RandomChoice([
            transforms_picture_overlay(),
            transforms_text_overlay(),
            transforms_torn_paper(),
        ])], p=prob_layout),
        transforms.RandomApply([transforms_erased_element()], p=prob_erased),
        transforms.RandomApply([transforms_add_gaussian_noise()], p=prob_noise),
        transforms.RandomApply([transforms_add_salt_and_pepper_noise()], p=prob_noise),
        transforms.RandomApply([transforms_dirty_rollers()], p=prob_noise),
        transforms.RandomApply([transforms_scribbles()], p=prob_stains),
        transforms.RandomApply([transforms_stains()], p=prob_stains),
        transforms.RandomApply([transforms_ink_bleed()], p=prob_stains),
        transforms.RandomApply([transforms_bleedthrough()], p=prob_stains),
        transforms.RandomApply([transforms.RandomChoice([
            transforms_crumpled_paper(),
            transforms_folded_paper()
        ])], p=prob_texture),
        transforms.RandomHorizontalFlip(p=prob_flip),
        transforms.RandomVerticalFlip(p=prob_flip),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=prob_blur),
        transforms.RandomApply([transforms_SepiaFilter()], p=prob_sepia)
    ])
 
    trial_name = f"c{int(prob_crop * 1000) % 1000}_" \
             f"h{int(prob_halftone * 1000) % 1000}_" \
             f"l{int(prob_layout * 1000) % 1000}_" \
             f"e{int(prob_erased * 1000) % 1000}_" \
             f"n{int(prob_noise * 1000) % 1000}_" \
             f"st{int(prob_stains * 1000) % 1000}_" \
             f"t{int(prob_texture * 1000) % 1000}_" \
             f"f{int(prob_flip * 1000) % 1000}_" \
             f"b{int(prob_blur * 1000) % 1000}_" \
             f"se{int(prob_sepia * 1000) % 1000}"
    
    # Entraîner SimCLR avec ces transformations 
    score = train_simclr(transform,trial_name)
    return score


# Lancer l'optimisation
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
