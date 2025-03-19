import sys
path = "."
if path not in sys.path:
    sys.path.insert(0, path)

import os
import json
from absolute_path import absolutePath
from PIL import Image
from tqdm import tqdm

corpus = 'guerre_illustree'
in_dir = absolutePath + 'corpus/guerre_illustree/retrieval/pages/'
out_dir = absolutePath + 'corpus/guerre_illustree/retrieval/photos/'

def crop(pages, photos):
    for page in tqdm(os.listdir(pages)):
        with open(pages + page, "r") as f:
            predictions = json.load(f)
        
        file = predictions['filepath']
        name = file.split("/")[-1]

        boxes = predictions['boxes']
        classes = predictions['pred_classes']
        ocr = predictions['ocr']

        im = Image.open(file)

        img_embeddings = []
        content_filepaths = []

        for i in range(len(boxes)):
            box = boxes[i]
            pred_class = classes[i]
            
            # if it's a headline, we skip the cropping
            if pred_class == 5:
                img_embeddings.append([])
                content_filepaths.append([])
                continue

            cropped = im.crop((box[0]*im.width, box[1]*im.height, box[2]*im.width, box[3]*im.height)).convert('RGB')
            cropped.save(photos + "jpg/" + name.replace(".jpg", "_" + str(i).zfill(3) + "_" + str(pred_class) + ".jpg"))
        
            with open(out_dir + "json/" + name.replace(".jpg", "_" + str(i).zfill(3) + "_" + str(pred_class) + ".json"), "w") as f:
                json.dump({"ocr": ocr[i]}, f)

crop(in_dir, out_dir)