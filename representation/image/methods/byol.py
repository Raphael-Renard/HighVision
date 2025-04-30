import sys
path = "../../.."
if path not in sys.path:
    sys.path.insert(0, path)


from tqdm import tqdm
import numpy as np
from data_retrieval import lipade_groundtruth
from data_retrieval.tools.data_loader import getDataLoader
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from data_retrieval import lipade_groundtruth
from clustering.clustering import getPredictionFromThreshold
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from byol_pytorch import BYOL
from torchvision import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

optimizerFunc = optim.Adam
temperature = 0.07
learningRate = 3e-4
batch_size = 512
workers = 2
# extract = 500
epochs = 100
corpus = "lipade_groundtruth"
resultsPath = "./results/distance/" + corpus + "/"


# Data
x,_,y = lipade_groundtruth.getDataset(mode = 'unique', uniform=True)

images = []
for i in range(len(x)):
    images.append(Image.open(x[i]).convert('RGB'))


x = np.array(x)
y = np.array(y)
images = np.array(images)


trainLoader = getDataLoader(images, None, None, False, batch_size, True, num_workers=2)


# Test
xSim,_,ySim = lipade_groundtruth.getDataset(mode = 'similar', uniform=True)
_,_,y_test = lipade_groundtruth.getDataset(mode="similar")

imagesSim = []
for i in range(len(xSim)):
    imagesSim.append(Image.open(xSim[i]).convert('RGB'))

testLoader = getDataLoader(imagesSim, None, None, False, batch_size, shuffle=False, num_workers=2)


# ## Transformations

from degradations.methods import transforms_floyd_steinberg_halftoning, transforms_atkinson_dithering, transforms_bayer_halftoning, transforms_add_gaussian_noise, transforms_add_salt_and_pepper_noise, transforms_dirty_rollers, transforms_ink_bleed, transforms_crumpled_paper, transforms_folded_paper, transforms_bleedthrough, transforms_scribbles, transforms_torn_paper, transforms_stains, transforms_erased_element, transforms_picture_overlay, transforms_text_overlay # Import your layout methods

class transforms_SepiaFilter(nn.Module):
    def __init__(self):
        super(transforms_SepiaFilter, self).__init__()

    def __call__(self, batch):
        sepia_filter = torch.tensor([[0.393, 0.769, 0.189],
                                     [0.349, 0.686, 0.168],
                                     [0.272, 0.534, 0.131]], device=batch.device)
        batch = torch.einsum('ijkl,mj->imkl', batch, sepia_filter)
        return batch.clamp(0, 1)


transform_degrad = transforms.Compose([
    transforms.RandomApply([
        transforms.RandomResizedCrop(size=256, scale=(1/2, 1), ratio=(1, 1))
    ], p=1/3),

    # halftone
    transforms.RandomApply([
        transforms.RandomChoice([
            transforms_floyd_steinberg_halftoning(),
            transforms_atkinson_dithering(),
            transforms_bayer_halftoning()
            ])
        ], p=0.2),

    # layout
    transforms.RandomApply([
            transforms.RandomChoice([
                transforms_picture_overlay(),
                transforms_text_overlay(),
                transforms_torn_paper(),
            ])
    ], p=0.2),

    # erased
    transforms.RandomApply([
        transforms_erased_element()
    ], p=0.1),

    # noise
    transforms.RandomApply([
            transforms.RandomChoice([
                transforms_add_gaussian_noise(),
                transforms_add_salt_and_pepper_noise(),
                transforms_dirty_rollers()
            ])
    ], p=0.2),

    # stains
    transforms.RandomApply([
        transforms.RandomChoice([
            transforms_scribbles(),
            transforms_stains(),
            transforms_ink_bleed(),
            transforms_bleedthrough(),
        ])
    ], p=0.3),

    # texture
    transforms.RandomApply([
        transforms.RandomChoice([
            transforms_crumpled_paper(),
            transforms_folded_paper(0.4),
        ])
    ], p=0.2),

    transforms.RandomHorizontalFlip(p=0.2),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.2),
    transforms.RandomApply([transforms_SepiaFilter()], p=0.2)
])





resnet = models.resnet50(pretrained=True)

learner = BYOL(
    resnet,
    image_size = 256,
    hidden_layer = 'avgpool',
    augment_fn = transform_degrad,
    augment_fn2 = transform_degrad
)

opt = torch.optim.Adam(learner.parameters(), lr=learningRate)

# path = "model_byol_degrad_learner.pth"
# learner.load_state_dict(torch.load(path))

for epoch in range(epochs):
    for i, batch in enumerate(tqdm(trainLoader, desc=f"Epoch {epoch}")):
        x = batch.to(device)
        loss = learner(x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average() # update moving average of target encoder

        # save improved network
        torch.save(resnet.state_dict(), 'model_byol_degrad.pth')
        torch.save(learner.state_dict(), 'model_byol_degrad_learner.pth')


def test_byol(model,name):
    xSim,_,ySim = lipade_groundtruth.getDataset(mode = 'similar', uniform=True)

    imagesSim = []
    for i in range(len(xSim)):
        imagesSim.append(Image.open(xSim[i]).convert('RGB'))

    testLoader = getDataLoader(imagesSim, None, None, False, batch_size, shuffle=False, num_workers=2)

    representations = []
    with torch.no_grad():
        for batch in testLoader:
            batch = model(batch.to(device),return_embedding = True)[1]
            for repr in batch.tolist():
                representations.append(repr)

    sim = cosine_similarity(representations, representations)

    distance = 1 - (sim+1)/2
    distance -= np.diag(distance)
    np.save(resultsPath + f"{name}.npy", distance)


test_byol(learner,'byol_degrad')