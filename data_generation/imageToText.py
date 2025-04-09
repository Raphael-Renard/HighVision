import sys
path = "."
if path not in sys.path:
    sys.path.insert(0, path)

from data_retrieval import lipade_groundtruth
from absolute_path import absolutePath
from lavis.models import load_model_and_preprocess
from PIL import Image
import torch
import csv

from parallelformers import parallelize

model, vis_processors, _ = load_model_and_preprocess(
    name="blip_caption", model_type="large_coco", is_eval=True, device="cpu"
)

parallelize(model, num_gpus=2, fp16=True, verbose="simple")

x,_,y = lipade_groundtruth.getDataset(mode="similar")
images = []
for i in range(len(x)):
    images.append(Image.open(x[i]).convert('RGB'))

image = images[0]
image = vis_processors["eval"](image).unsqueeze(0)
print(model.generate({"image": image, "prompt": "Question: What is represented in this photograph? Only talk about the content, not the style. Answer:"}))