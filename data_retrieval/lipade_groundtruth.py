import sys
path = "."
if path not in sys.path:
    sys.path.insert(0, path)

import os
import pandas as pd
from absolute_path import absolutePath

def getDataset(mode, check=False):
    # Paths
    corpusPath = absolutePath + 'corpus/lipade_groundtruth/'

    similarPath = os.path.join(corpusPath, 'similar/')
    uniquePath = os.path.join(corpusPath, 'unique/')
    groundtruthPath = os.path.join(corpusPath, 'groundtruth.xlsx')

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
                labels.append(g)
        labelPadding = len(groups)

    if mode in ["unique", "all"] :
        # Load unique images
        partial_paths = os.listdir(uniquePath)
        for index in range(len(partial_paths)):
            path = os.path.join(uniquePath, partial_paths[index])
            paths.append(path)
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

    return paths, None, labels # Images / Metadata / Groundtruth labels

# Display
if __name__ == '__main__':
    x,_,_ = getDataset('all')
    print(len(x), 'images')
    x,_,_ = getDataset('similar')
    print(len(x), 'similar images')
    x,_,_ = getDataset('unique')
    print(len(x), 'unique images')