import sys
path = "."
if path not in sys.path:
    sys.path.insert(0, path)

import os

from data_retrieval import lipade_groundtruth
from absolute_path import absolutePath
from lavis.models import load_model_and_preprocess
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import csv

corpus = "lipade_groundtruth"
device = "cuda" if torch.cuda.is_available() else "cpu"

model, vis_processors, _ = load_model_and_preprocess(
    name="blip_caption", model_type="large_coco", is_eval=True, device="cpu"
)
model.half().to(device)

x_s,_,y_s = lipade_groundtruth.getDataset(mode="similar")

x_u,m_u,_ = lipade_groundtruth.getDataset(mode="unique")
is_recto = np.array(m_u[2])
x_u = np.array(x_u)[is_recto]

path = absolutePath + 'data_generation/generated/' + corpus + '.csv'

if os.path.exists(path):
    with open(path, mode='r') as infile:
        reader = csv.reader(infile, delimiter=';')
        dict = {(rows[0],rows[1]): rows[2] for rows in reader}
else:
    dict = {}

for x in [x_u, x_s]:
    for i in tqdm(range(len(x))):
        image = Image.open(x[i]).convert('RGB')
        filename = '/'.join(x[i].split('/')[-2:])

        if (filename, "") in dict:
            continue

        image = vis_processors["eval"](image).unsqueeze(0).to(device, torch.float16)
        generated_text = model.generate({"image": image})
        dict[(filename, "")] = generated_text[0].strip()

        with open(path, 'w') as outfile:
            res_dict = csv.writer(outfile, delimiter=';')
            kv = list(dict.items())
            for key, value in kv:
                res_dict.writerow([key[0], key[1], value])