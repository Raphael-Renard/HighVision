import sys
path = "."
if path not in sys.path:
    sys.path.insert(0, path)

import os
from absolute_path import absolutePath
from data_retrieval.tools.british_newspaper_archive_retrieval import britishNewspaperArchiveExplorer
import datetime
from pdf2image import convert_from_path
import numpy as np
import pytesseract
from PIL import Image
from tqdm import tqdm

# Scrapping
def downloadDataset(mode = 0):
    issuesRetrieval = mode == 0
    pagesRetrieval = mode == 1
    download = mode == 2

    corpusPath = absolutePath + 'corpus/daily_mirror/'
    explorer = britishNewspaperArchiveExplorer("daily mirror", 1914, 1920, corpusPath + "bna_downloads/")

    if issuesRetrieval:
        with open("corpus/daily_mirror/bna_download_index.txt") as f:
            startIndex = int(f.readline())

        explorer.getIssues(startIndex, "corpus/daily_mirror/bna_issue_index.txt")
        with open("corpus/daily_mirror/bna_issues.txt", "w") as f:
            for href in explorer.issues:
                f.write(href + "\n")

    if pagesRetrieval:
        with open("corpus/daily_mirror/bna_page_index.txt") as f:
            startIndex = int(f.readline())

        explorer.getPages("corpus/daily_mirror/bna_issues.txt", startIndex, "corpus/daily_mirror/bna_page_index.txt")
        with open("corpus/daily_mirror/bna_pages.txt", "a") as f:
            for href in explorer.pages:
                f.write(href + "\n")

    if download:
        startIndex = len(os.listdir(os.path.join(absolutePath, corpusPath + "bna_downloads/")))
        explorer.getPdf("corpus/daily_mirror/bna_pages.txt", startIndex, "corpus/daily_mirror/bna_download_index.txt")

def pdfToImage():
    pdfDir = absolutePath + 'corpus/daily_mirror/bna_downloads/'
    jpgDir = absolutePath + 'corpus/daily_mirror/pages/'

    for file in tqdm(os.listdir(pdfDir)):
        try:
            name = f"Daily_mirror_{file[11:15]}_{file[15:17]}_{file[17:19]}_p{file[26:28]}.jpg"
            if name not in os.listdir(jpgDir):
                image = np.array(convert_from_path(pdfDir + file)[0])
                whiteLineFilter = np.where(np.any(image < 250, axis=(1, 2)))[0]
                image = image[whiteLineFilter[0] : whiteLineFilter[-1] + 1]
                Image.fromarray(image).save(jpgDir + name, 'JPEG')
        except:
            with open("corpus/daily_mirror/error.txt", "a") as f:
                f.write(file + "\n")

def getMetaData():
    jpgDir = absolutePath + 'corpus/daily_mirror/pages/'
    xmlDir = absolutePath + 'corpus/daily_mirror/xml/'

    # language : eng / fra
    custom_config = r'-l eng --oem 1 --psm 1 -c tessedit_create_alto=1'

    for im in tqdm(os.listdir(jpgDir)):
        xml = im.split(".")[0] + ".xml"
        if xml not in os.listdir(xmlDir):
            image = Image.open(jpgDir + im)
            alto_xml = pytesseract.image_to_alto_xml(image, config=custom_config).decode('utf-8')

            with open(xmlDir + xml, 'w', encoding='utf-8') as f:
                f.write(alto_xml)

# Retrieval
def getDataset():
    # Paths
    corpusPath = absolutePath + 'corpus/daily_mirror/pages/'

    # Load
    paths = []
    metadata = []
    for path in os.listdir(corpusPath):
        paths.append(path)
        print(path)
        date = datetime.date()
        metadata.append({'date' : date})

    return paths, metadata, None # Images / Metadata / Groundtruth labels

# Display
if __name__ == '__main__':
    # downloadDataset(0)
    # downloadDataset(1)
    # downloadDataset(2)
    pdfToImage()
    # getMetaData()
    # x,_,_ = getDataset()
    # print(len(x), 'images')