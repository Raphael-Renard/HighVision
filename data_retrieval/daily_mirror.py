import sys
path = "."
if path not in sys.path:
    sys.path.insert(0, path)

import os
from absolute_path import absolutePath
from tools.british_newspaper_archive_retrieval import britishNewspaperArchiveExplorer
import datetime

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

# Retrieval
def getDataset():
    # Paths
    corpusPath = absolutePath + 'corpus/lipade_groundtruth/bna_downloads/'

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
    downloadDataset(2)
    # x,_,_ = getDataset()
    # print(len(x), 'images')