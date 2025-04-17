import sys
path = "."
if path not in sys.path:
    sys.path.insert(0, path)

import os

root = os.environ['DSDIR'] + '/HuggingFace_Models/'

from data_retrieval import lipade_groundtruth
from absolute_path import absolutePath
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import numpy as np
import torch
import csv

corpus = "lipade_groundtruth"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
model = Blip2ForConditionalGeneration.from_pretrained(root + "Salesforce/blip2-flan-t5-xxl", device_map="auto")

x_s,_,y_s = lipade_groundtruth.getDataset(mode="similar")

x_u,m_u,_ = lipade_groundtruth.getDataset(mode="unique")
is_recto = np.array(m_u[2])
x_u = np.array(x_u)[is_recto]

prompts = ["EMPTY", "A photograph representing ", "a newspaper clipping from the early 1900s showing ", "an old newspaper article with ", "a black and white photo of ", "Question: What is represented in this photograph? Answer: ", "Question: What is represented in this photograph? Only talk about the content, not the style. Answer: "]

path = absolutePath + 'data_generation/generated/' + corpus + "/"

for x in [x_u, x_s]:
    for i in tqdm(range(len(x))):
        for prompt in prompts:
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

            if prompt == "EMPTY":
                inputs = processor(image, "", return_tensors="pt").to("cuda")
                out = model.generate(**inputs)
                generated_text = processor.decode(out[0], skip_special_tokens=True)

            else:
                inputs = processor(image, prompt, return_tensors="pt").to("cuda")
                out = model.generate(**inputs)
                generated_text = processor.decode(out[0], skip_special_tokens=True)

            print(generated_text)
            break
            dict[prompt] = generated_text[0].strip()

            with open(path + filename, 'w') as outfile:
                res_dict = csv.writer(outfile, delimiter=';')
                kv = list(dict.items())
                for key, value in kv:
                    res_dict.writerow([key, value])