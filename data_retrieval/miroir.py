import sys
path = "."
if path not in sys.path:
    sys.path.insert(0, path)

from absolute_path import absolutePath
from data_retrieval.tools.gallica_retrieval import download

ark = 'ark:/12148/cb34419118b'

download(ark, absolutePath + '/corpus/le_miroir')

print("Finished !")