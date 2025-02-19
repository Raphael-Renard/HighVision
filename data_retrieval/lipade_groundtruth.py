import sys
path = "."
if path not in sys.path:
    sys.path.insert(0, path)

import os
import pandas as pd
from absolute_path import absolutePath

def getDataset(check=False):
    # Paths
    corpusPath = absolutePath + 'corpus/lipade_groundtruth/'

    imagesPath = os.path.join(corpusPath, 'images/')
    groundtruthPath = os.path.join(corpusPath, 'groundtruth.xlsx')

    # Load
    groundtruth = pd.read_excel(groundtruthPath, header=None)
    groups = [[image for image in group if type(image) is str] for group in groundtruth.values.tolist()] # Remove NaN values
    paths = []
    labels = []
    for g in range(len(groups)):
        for i in range(len(groups[g])):
            if groups[g][i].endswith('.tif') or groups[g][i].endswith('.jpg'):
                groups[g][i] = os.path.join(imagesPath, groups[g][i])
            else:
                groups[g][i] = os.path.join(imagesPath, groups[g][i] + '.jpg')
            paths.append(groups[g][i])
            labels.append(g)

    # Check
    if check:
        for g in range(len(groups)):
            for i in range(len(groups[g])):
                if not os.path.exists(groups[g][i]):
                    print(g,i,groups[g][i].split("/")[-1],"does not exist")
        for path in os.listdir(imagesPath):
            if not any(os.path.join(imagesPath, path) in group for group in groups):
                print(path, "is not in any group")

    return paths, None, labels # Images / Metadata / Groundtruth labels

# Display
if __name__ == '__main__':
    x,_,_ = getDataset()
    print(len(x), 'images')