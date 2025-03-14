import sys
path = "."
if path not in sys.path:
    sys.path.insert(0, path)

import os
from absolute_path import absolutePath

imagePath = lambda yearmonth, name : f"corpus/guerre_illustree/data/{yearmonth}/pages/jpg/{name}.jpg"
xmlPath = lambda yearmonth, name : f"corpus/guerre_illustree/data/{yearmonth}/pages/xml/{name}.xml"
outPath = lambda name : f"corpus/guerre_illustree/retrieval/{name}.json"

pages = []
for yearmonth in os.listdir(absolutePath + "corpus/guerre_illustree/data"):
    for page in os.listdir(absolutePath + "corpus/guerre_illustree/data/" + yearmonth + "/pages/jpg/"):
        pageName = page.split(".")[0]
        pages.append(imagePath(yearmonth, pageName))

import torch
import json
import cv2
import numpy as np
from tqdm import tqdm 
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

torch.cuda.empty_cache()

def generate_predictions(filepaths, id):
    #with torch.cuda.device(id):
        # sets up model for process
        setup_logger()
        cfg = get_cfg()
        cfg.merge_from_file(absolutePath + "layout_segmentation/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

        # ("Illustration/Photograph", "Photograph", "Comics/Cartoon", "Editorial Cartoon", "Map", "Headline", "Ad")
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7

        model = build_model(cfg)

        # https://news-navigator.labs.loc.gov/model_weights/model_final.pth
        DetectionCheckpointer(model).load(absolutePath + "layout_segmentation/weight/model_final.pth")
        model.train(False) 

        inputs = []
        dimensions = []

        for file in tqdm(filepaths):
            image = cv2.imread(file)
            height, width, _ = image.shape
            dimensions.append([width, height])
            image = np.transpose(image,(2,0,1))
            image_tensor = torch.from_numpy(image)
            inputs.append({"image": image_tensor})

        outputs = model(inputs)
        predictions = {}

        for i in tqdm(range(len(filepaths))):
            predictions["filepath"] = filepaths[i]
            predictions["pub_date"] = predictions["filepath"][-13:-7]
            predictions["page_seq_num"] = predictions["filepath"][-6:-4]

            boxes = outputs[i]["instances"].get_fields()["pred_boxes"].to("cpu").tensor.tolist()
            normalized_boxes = []
            width = dimensions[i][0]
            height = dimensions[i][1]
            for box in boxes:
                normalized_box = (box[0]/float(width), box[1]/float(height), box[2]/float(width), box[3]/float(height))
                normalized_boxes.append(normalized_box)

            predictions["boxes"] = normalized_boxes
            predictions["scores"] = outputs[i]["instances"].get_fields()["scores"].to("cpu").tolist()
            predictions["pred_classes"] = outputs[i]["instances"].get_fields()["pred_classes"].to("cpu").tolist()

            with open(outPath(filepaths[i].split("/")[-1][:-4]), "w") as fp:
                json.dump(predictions, fp)

N = len(pages)
for i in range(N):
    l = len(pages) // N
    generate_predictions(pages[i*l:(i+1)*l], 1)