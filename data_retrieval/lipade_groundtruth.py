import sys
path = "."
if path not in sys.path:
    sys.path.insert(0, path)

import os
import pandas as pd
import xml.etree.ElementTree as ET
import langdetect
from absolute_path import absolutePath
from deep_translator import GoogleTranslator

def getMetadata(metaRow):
    meta = []
    meta.append(metaRow['name'].iloc[0])
    data = metaRow['description']
    if not data.isnull().iloc[0]:
        meta.append(data.iloc[0])
    data = metaRow['legende']
    if not data.isnull().iloc[0]:
        meta.append(data.iloc[0])
    data = metaRow['legende_2']
    if not data.isnull().iloc[0] and "COPYRIGHT" not in data.iloc[0]:
        meta.append(data.iloc[0])
    data = metaRow['xml']
    if not data.isnull().iloc[0]:
        root = ET.parse(absolutePath + "corpus/lipade_groundtruth/xml/" + data.iloc[0]).getroot()
        for c in root.find('notice').find('record').find('metadata')[0]:
            if "subject" in c.tag:
                if "Borea" not in c.text and "Caneva" not in c.text:
                    meta.append(c.text)

    traductor = GoogleTranslator(source="auto", target="fr")
    for i in range(len(meta) - 1):
        data = meta[i+1]
        try:
            lang = langdetect.detect(data)
        except:
            continue
        if lang == 'en':
            translation = traductor.translate(data)
            meta[i+1] = translation

    return ";".join(meta)

def getDataset(mode, check=False, uniform=False, writeMeta=False):
    corpusFolder = 'mini_corpus/' if uniform else 'corpus/'

    # Paths
    corpusPath     = absolutePath + corpusFolder + 'lipade_groundtruth/'
    trueCorpusPath = absolutePath + 'corpus/'    + 'lipade_groundtruth/'

    similarPath = os.path.join(corpusPath, 'similar/')
    uniquePath  = os.path.join(corpusPath, 'unique/')
    groundtruthPath = os.path.join(trueCorpusPath, 'groundtruth.xlsx')
    metadataPath = os.path.join(trueCorpusPath, 'metadata.xlsx')
    selectedMetadataPath = os.path.join(trueCorpusPath, 'selectedMetadata.csv')

    if writeMeta:
        meta = []
        metadata = pd.read_csv(metadataPath)
    else:
        with open(selectedMetadataPath, "r") as f:
            meta = [" ; ".join(line.rstrip().split(";")[1:]) for line in f.readlines()]

    paths = []
    labels = []
    labelPadding = 0
    if mode in ["similar", "all"] :
        # Load similar images
        groundtruth = pd.read_excel(groundtruthPath, header=None)
        groups = [[image for image in group if type(image) is str] for group in groundtruth.values.tolist()] # Remove NaN values
        for g in range(len(groups)):
            for i in range(len(groups[g])):
                if groups[g][i].endswith('.tif') or groups[g][i].endswith('.jpg'):
                    groups[g][i] = os.path.join(similarPath, groups[g][i])
                else:
                    groups[g][i] = os.path.join(similarPath, groups[g][i] + '.jpg')
                paths.append(groups[g][i])
                if writeMeta:
                    meta.append(getMetadata(metadata.loc[metadata['name'] == groups[g][i].split("/")[-1]]))
                labels.append(g)
        labelPadding = len(groups)

    if mode in ["unique", "all"] :
        # Load unique images
        partial_paths = os.listdir(uniquePath)
        for index in range(len(partial_paths)):
            path = os.path.join(uniquePath, partial_paths[index])
            paths.append(path)
            if writeMeta:
                meta.append(getMetadata(metadata.loc[metadata['name'] == path.split("/")[-1]]))
            labels.append(labelPadding + index)

    # Check groundtruth paths coherence
    if check:
        for g in range(len(groups)):
            for i in range(len(groups[g])):
                if not os.path.exists(groups[g][i]):
                    print(g,i,groups[g][i].split("/")[-1],"does not exist")
        for path in os.listdir(similarPath):
            if not any(os.path.join(similarPath, path) in group for group in groups):
                print(path, "is not in any group")

    if writeMeta:
        with open(selectedMetadataPath, 'w') as f:
            f.writelines([line.replace("\n", "") + "\n" for line in meta])
    else:
        l = len(os.listdir(similarPath))
        if mode == 'similar':
            meta = meta[:l]
        elif mode == 'unique':
            meta = meta[l:]

    return paths, meta, labels # Images / Metadata / Groundtruth labels

# Display
if __name__ == '__main__':
    x,_,_ = getDataset('all')
    print(len(x), 'images')
    x,_,_ = getDataset('similar')
    print(len(x), 'images')
    x,_,_ = getDataset('unique')
    print(len(x), 'images')