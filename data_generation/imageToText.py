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
import torch
import csv

from parallelformers import parallelize

corpus = "lipade_groundtruth"

os.environ["TORCH_HOME"] = "/lustre/fsstor/projects/rech/dvj/umv31gt/models/lavis"

model, vis_processors, _ = load_model_and_preprocess(
    name="blip_caption", model_type="large_coco", is_eval=True, device="cpu"
)

parallelize(model, fp16=False, num_gpus=4, verbose="simple")

x_s,_,y_s = lipade_groundtruth.getDataset(mode="similar")

x_u,m_u,_ = lipade_groundtruth.getDataset(mode="unique")
is_recto = np.array(m_u[2])
x_u = np.array(x_u)[is_recto]

prompts = ["", "A photograph representing ", "a newspaper clipping from the early 1900s showing ", "an old newspaper article with ", "a black and white photo of ", "Question: What is represented in this photograph? Answer: ", "Question: What is represented in this photograph? Only talk about the content, not the style. Answer:"]

path = absolutePath + 'data_generation/generated/' + corpus + "/"

for x in [x_s, x_u]:
    for prompt in prompts:
        for i in range(len(x)):
            image = Image.open(x[i]).convert('RGB')
            filename = x[i].split('/')[-1].split('.')[0] + '.csv'
            if os.path.exists(path + filename):
                with open(path + filename, mode='r') as infile:
                    reader = csv.reader(infile, delimiter=';')
                    dict = {rows[0]: rows[1] for rows in reader}
            else:
                dict = {}

            if prompt in dict:
                continue

            print(filename)

            image = vis_processors["eval"](image).unsqueeze(0)
            if prompt == "":
                generated_text = model.generate({"image": image})
            else:
                generated_text = model.generate({"image": image, "prompt": prompt})
            dict[prompt] = generated_text[0].strip()

            with open(path + filename, 'w') as outfile:
                res_dict = csv.writer(outfile, delimiter=';')
                kv = list(dict.items())
                for key, value in kv:
                    res_dict.writerow([key, value])