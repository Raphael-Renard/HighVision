import sys
path = "."
if path not in sys.path:
    sys.path.insert(0, path)

from data_retrieval import lipade_groundtruth
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os

root = os.environ['DSDIR'] + '/HuggingFace_Models/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
corpus = "lipade_groundtruth"
distancePath = "../results/distance/" + corpus + "/"
rawPath = "../results/raw/" + corpus + "/"
weightsPath = "../results/weights/" + corpus + "/"
model_save_path = "clip.pth"
batch_size = 32
num_epochs = 3

# Model
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained(root + "openai/clip-vit-base-patch32").to(device)
torch.save(model.state_dict(), weightsPath + f"pretrained_{model_save_path}")

# Images
x,m,_ = lipade_groundtruth.getDataset(mode="unique")
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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.image_paths[idx], self.captions[idx], idx

def semantic_matching_loss(image_embeds, text_embeds, M, tau=0.07):
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    logits = (image_embeds @ text_embeds.T) / tau
    pred_probs = F.log_softmax(logits, dim=-1)

    target_probs = F.softmax(M.to(device), dim=-1)
    loss = -torch.sum(target_probs * pred_probs) / image_embeds.size(0)
    return loss

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)

captions = images_per_captions["a black and white photo of "]

train_images, val_images, train_captions, val_captions = train_test_split(
    x, captions, test_size=0.1, random_state=42
)

train_loader = DataLoader(CustomDataset(train_images, train_captions), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(CustomDataset(val_images, val_captions), batch_size=batch_size, shuffle=False)

def evaluate_model(model, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for paths, captions, indices in val_loader:
            images = []
            for path in paths:
                image = Image.open(path).convert("RGB")
                images.append(image)
            captions = list(captions)
            inputs = processor(text=captions, images=images, return_tensors="pt", padding=True, truncation=True)
            inputs = inputs.to(device)
            outputs = model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            M = torch.eye(len(indices)).float() # Similarity matrix
            
            loss = semantic_matching_loss(image_embeds, text_embeds, M)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss

for epoch in range(num_epochs):
    model.train()
    for batch_id, (paths, captions, indices) in enumerate(train_loader):
        images = []
        for path in paths:
            image = Image.open(path).convert("RGB")
            images.append(image)
        captions = list(captions)
        inputs = processor(text=captions, images=images, return_tensors="pt", padding=True, truncation=True)
        inputs = inputs.to(device)
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        M = torch.eye(len(indices)).float() # Similarity matrix

        loss = semantic_matching_loss(image_embeds, text_embeds, M)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.save(model.state_dict(), weightsPath + f"c1_e{epoch}_{model_save_path}")
    val_loss = evaluate_model(model, val_loader)
    print(f"[Epoch {epoch}] Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")