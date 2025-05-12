# HighVision
Identification of image circulation in large collections of historical photographs.

## 1- Tests
### GPU tests
## 2- Corpus
### Bain
### Cherau
### Chronic
### Forbin
### Gallica:Full
### Gallica:Photo
### Guerre Illustree
### Groundtruth (LIPADE)
### Presse (LIPADE)
### Rol
### Sur le vif
## 3- Data Retrieval
## 4- Layout Segmentation
## 5- Degradations
### Overview
Simulate various types of degradations commonly found in historical photographs and printed materials. 

### Features
- **Halftoning Techniques**:
  - Atkinson dithering
  - Floyd-Steinberg dithering
  - Bayers threshold halftoning
- **Noise Addition**:
  - Gaussian noise
  - Salt-and-pepper noise
  - dirty rollers
- **Paper Imperfections**:
  - bleedthrough
  - contrast
  - crumpled paper effect
  - folded paper effect
  - ink bleed
  - book crease / picture cut in 2 parts (pliure_livre)
  - scribbles
  - stains
  - torn paper
  - blue tone
  - faded effect
- **Layout**:
  - corner overlay (picture_overlay)
  - text overlay
  - change frame shape (cadre)
  - patchwork 
  - photomontage
  - rectangular frame around the picture (encadre_rectangle)
- **Human made corrections**:
  - object/element erased
  - drawing effect
  - paint effect

## 5 bis- Data Generation
## 6- Representation
### Image representation
SimCLR, Sift, Byol

### Multimodal representation
Clip, Flava, Blip
## 7- Clustering
### Generation of clusters
### Evaluation of clusters
## 8- Circulation
