import sys
path = "."
if path not in sys.path:
    sys.path.insert(0, path)

import os
import pandas as pd
from absolute_path import absolutePath

def getDataset(mode, uniform=False):
    corpusFolder = 'mini_corpus/' if uniform else 'corpus/'

    # Paths
    corpusPath     = absolutePath + corpusFolder + 'forbin/'
    trueCorpusPath = absolutePath + 'corpus/'    + 'forbin/'

    if mode == 'enveloppe':
        paths = {}
        metadata = {}
    else:
        paths = []
        metadata = []

    for excel in os.listdir(trueCorpusPath):
        if excel.endswith('.xlsx'):
            df = pd.read_excel(trueCorpusPath + excel)
            df[['Titre Enveloppe', 'Titre Pochette']] = df[['Titre Enveloppe', 'Titre Pochette']].ffill()

            df = df.dropna(subset=['Image']) # Retrait des categories
            df = df.dropna(subset=['Face'])  # Retrait de la vue technique

            # Fusion des recto/verso
            df['Key'] = df['Image'].str.rsplit('_', n=1).str[0]
            recto_df = df[df['Face'] == 'recto'].rename(columns={'Image': 'Recto'}).drop(columns=['Face'])
            verso_df = df[df['Face'] == 'verso'].rename(columns={'Image': 'Verso'}).drop(columns=['Face'])
            df = pd.merge(recto_df, verso_df, on=['Dossier', 'Conditionnement', 'Titre Enveloppe', 'Titre Pochette', 'Key'], how='left')
        
            for _, row in df.iterrows():
                recto_path = corpusPath + 'Boites/' + row['Dossier'] + '/' + row['Recto'] + '.jpg'

                if mode == 'image':
                    paths.append(recto_path)

                    meta = {
                        "enveloppe": row["Titre Enveloppe"],
                        "pochette": row["Titre Pochette"],
                        "conditionnement": row["Conditionnement"],
                        "verso": None
                    }
                    if pd.notna(row["Verso"]):
                        verso_path = corpusPath + 'Boites/' + row['Dossier'] + '/' + row['Verso'] + '.jpg'
                        meta["verso"] = verso_path

                    metadata.append(meta)
                elif mode == 'enveloppe':
                    enveloppe = row["Titre Enveloppe"]
                    pochette = row["Titre Pochette"]
                    if enveloppe not in paths:
                        paths[enveloppe] = {}
                        metadata[enveloppe] = {}
                    if pochette not in paths[enveloppe]:
                        paths[enveloppe][pochette] = []
                        metadata[enveloppe][pochette] = []
                    paths[enveloppe][pochette].append(recto_path)

                    meta = {
                        "verso": None,
                        "conditionnement": row["Conditionnement"]
                    }
                    if pd.notna(row["Verso"]):
                        verso_path = corpusPath + row['Dossier'] + '/' + row['Verso'] + '.jpg'
                        meta["verso"] = verso_path

                    metadata[enveloppe][pochette].append(meta)

    return paths, metadata, None # Images / Metadata / Groundtruth labels

# Display
if __name__ == '__main__':
    x,m_,_ = getDataset('image')
    print(len(x), 'images')
    x,_,_ = getDataset('enveloppe')
    print(len(x), 'enveloppes')