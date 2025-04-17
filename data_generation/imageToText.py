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

path = absolutePath + 'data_generation/generated/' + corpus + "/"

for x in [x_u, x_s]:
    for i in tqdm(range(len(x))):
        image = Image.open(x[i]).convert('RGB')
        filename = x[i].split('/')[-1].split('.')[0] + '.csv'
        if os.path.exists(path + filename):
            with open(path + filename, mode='r') as infile:
                reader = csv.reader(infile, delimiter=';')
                dict = {rows[0]: rows[1] for rows in reader}
        else:
            dict = {}

        if "" in dict:
            continue

        image = vis_processors["eval"](image).unsqueeze(0).to(device, torch.float16)
        generated_text = model.generate({"image": image})
        dict[""] = generated_text[0].strip()

        with open(path + filename, 'w') as outfile:
            res_dict = csv.writer(outfile, delimiter=';')
            kv = list(dict.items())
            for key, value in kv:
                res_dict.writerow([key, value])