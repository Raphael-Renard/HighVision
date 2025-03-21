import sys
path = "."
if path not in sys.path:
    sys.path.insert(0, path)

import os
import fnmatch
import json
from absolute_path import absolutePath
from lxml import etree as ET

def retrieveOcrFromXml(xmlPath, boxes):
    ocr = [[] for i in range(len(boxes))]

    parser = ET.XMLParser()
    tree = ET.parse(xmlPath, parser)
    root = tree.getroot()
    prefix = root.tag.split('}')[0] + '}'
    layout = root.find(prefix + 'Layout')
    page = layout.find(prefix + 'Page')
    print_space = page.find(prefix + 'PrintSpace')
    if print_space is None:
        return ocr

    text_boxes =  [textblock for textblock in print_space.iterchildren(prefix + "TextBlock")]
    
    # gets page height and page width in inch1200 units
    page_width_inch = int(page.attrib['WIDTH'])
    page_height_inch = int(page.attrib['HEIGHT'])

    # sets conversion to normalized coordinates for comparison between METS/ALTO and predicted boxes
    W_CONVERSION = 1./float(page_width_inch)
    H_CONVERSION = 1./float(page_height_inch)

    if page_width_inch == 0 or page_height_inch == 0:
        return ocr

    # we now iterate over each bounding box
    for i in range(0, len(boxes)):
        bounding_box = boxes[i]

        # we then iterate over each text box
        for text_box in text_boxes:
            box_w1 = int(float(text_box.attrib["HPOS"]))
            box_h1 = int(float(text_box.attrib["VPOS"]))
            box_w2 = box_w1 + int(float(text_box.attrib["WIDTH"]))
            box_h2 = box_h1 + int(float(text_box.attrib["HEIGHT"]))
            
            # if the text box and bounding box do not intersect, we skip (as no text will overlap in smaller units)
            if box_w2*W_CONVERSION < bounding_box[0] and box_h2*H_CONVERSION < bounding_box[1]:
                continue
            if box_w1*W_CONVERSION > bounding_box[0] + bounding_box[2] and box_h2*H_CONVERSION < bounding_box[1]:
                continue
            if box_w2*W_CONVERSION < bounding_box[0] and box_h1*H_CONVERSION > bounding_box[1] + bounding_box[3]:
                continue
            if box_w1*W_CONVERSION > bounding_box[0] + bounding_box[2] and box_h1*H_CONVERSION > bounding_box[1] + bounding_box[3]:
                continue
                
            # we then iterate over the text lines in each box
            for text_line in text_box.iterchildren(prefix + 'TextLine'):
                line_w1 = int(float(text_box.attrib["HPOS"]))
                line_h1 = int(float(text_box.attrib["VPOS"]))
                line_w2 = line_w1 + int(float(text_box.attrib["WIDTH"]))
                line_h2 = line_h1 + int(float(text_box.attrib["HEIGHT"]))

                # if the text box and bounding box do not intersect, we skip (as no text will overlap in smaller units)
                if line_w2*W_CONVERSION < bounding_box[0] and line_h2*H_CONVERSION < bounding_box[1]:
                    continue
                if line_w1*W_CONVERSION > bounding_box[0] + bounding_box[2] and line_h2*H_CONVERSION < bounding_box[1]:
                    continue
                if line_w2*W_CONVERSION < bounding_box[0] and line_h1*H_CONVERSION > bounding_box[1] + bounding_box[3]:
                    continue
                if line_w1*W_CONVERSION > bounding_box[0] + bounding_box[2] and line_h1*H_CONVERSION > bounding_box[1] + bounding_box[3]:
                    continue
                
                # we now iterate over every string in each line (each string is separated by whitespace)
                for string in text_line.iterchildren(prefix + 'String'):
            
                    w1 = int(float(string.attrib["HPOS"]))
                    h1 = int(float(string.attrib["VPOS"]))
                    w2 = w1 + int(float(string.attrib["WIDTH"]))
                    h2 = h1 + int(float(string.attrib["HEIGHT"]))

                    # checks if the text appears within the bounding box & extra tolerance for words that are clipped
                    WIDTH_TOLERANCE, HEIGHT_TOLERANCE = 0,0
                    if w1*W_CONVERSION > bounding_box[0] - WIDTH_TOLERANCE:
                        if w2*W_CONVERSION < bounding_box[2] + WIDTH_TOLERANCE:
                            if h1*H_CONVERSION > bounding_box[1] - HEIGHT_TOLERANCE:
                                if h2*H_CONVERSION < bounding_box[3] + HEIGHT_TOLERANCE:
                                    ocr[i].append(string.attrib["CONTENT"])

    return ocr

def getDataset(mode, uniform=False, jsonMetadata=True):
    corpusFolder = 'mini_corpus/' if uniform else 'corpus/'

    corpusPath     = absolutePath + corpusFolder + 'guerre_illustree/data/'
    trueCorpusPath = absolutePath + 'corpus/'    + 'guerre_illustree/data/'

    paths = []
    metadata = []

    if mode == "page":
        for date in os.listdir(corpusPath):
            for page in os.listdir(os.path.join(corpusPath, date, 'pages/jpg/')):
                paths.append(os.path.join(corpusPath, date, 'pages/jpg/', page))
                metadata.append({})
                prefix = page.split('.')[0]
                metadata[-1]['date'] = date
                metadata[-1]['xml'] = os.path.join(corpusPath, date, 'pages/xml/', prefix + '.xml')
                metadata[-1]['datexml'] = os.path.join(corpusPath, date, date + '.xml')
                photos = []
                for photo in os.listdir(os.path.join(corpusPath, date, 'photos/jpg/')):
                    if fnmatch.fnmatch(photo, f'{prefix}_*'):
                        photos.append(os.path.join(corpusPath, date, 'photos/jpg/', photo))
                metadata[-1]['photos'] = photos

                if jsonMetadata:
                    jsonPath = absolutePath + 'corpus/guerre_illustree/retrieval/pages/'
                    for jsonFile in os.listdir(jsonPath):
                        with open(jsonPath + jsonFile, "r") as f:
                            jsonInfos = json.load(f)
                        metadata[-1]['pred_classes'] = jsonInfos['pred_classes']
                        metadata[-1]['boxes'] = jsonInfos['boxes']
                        if 'ocr' not in jsonInfos:
                            ocr = retrieveOcrFromXml(metadata[-1]['xml'], metadata[-1]['boxes'])
                            jsonInfos['ocr'] = ocr
                            with open(jsonPath + jsonFile, "w") as f:
                                json.dump(jsonInfos, f)
                        metadata[-1]['ocr'] = jsonInfos['ocr']


    elif mode == "photo":
        for date in os.listdir(corpusPath):
            for photo in os.listdir(os.path.join(corpusPath, date, 'photos/jpg/')):
                paths.append(os.path.join(corpusPath, date, 'photos/jpg/', photo))
                metadata.append({})
                prefix = '_'.join(photo.split('_')[:-1])
                metadata[-1]['date'] = date
                metadata[-1]['page'] = os.path.join(corpusPath, date, 'pages/jpg/', prefix + '.jpg')

    return paths, metadata, None # Images / Metadata / Groundtruth labels

# Display
if __name__ == '__main__':
    x,_,_ = getDataset('page')
    print(len(x), 'pages')
    x,_,_ = getDataset('photo')
    print(len(x), 'photos')