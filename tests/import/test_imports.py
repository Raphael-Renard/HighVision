# Must be executed from the root of the project

import sys
path = "."
if path not in sys.path:
    sys.path.insert(0, path)

from data_retrieval.lipade_groundtruth import getDataset
print("OK")