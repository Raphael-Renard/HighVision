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
  (Different methods to do the same thing)
  - **atkinson**: Atkinson method for dithering
  - **floyd_steinberg**: Floyd-Steinberg method for dithering
  - **bayers_threshold**: Bayers threshold for halftoning
- **Noise Addition**:
  - **gaussian_noise**: gaussian noise
  - **salt_and_pepper**: salt-and-pepper noise (black and white pixels)
  - **dirty_rollers**: parallel lines left by a dirty printing press
- **Paper Imperfections**:
  - **bleedthrough**: the ink from the other side of the paper bled through the paper
  - **contrast**: the black and white are hightened
  - **crumpled_paper**: crumpled paper
  - **folded_paper**: folded paper
  - **ink_bleed**: the ink has smudged
  - **pliure_livre**: book crease / picture cut in 2 parts
  - **scribbles**: pen marks
  - **stains**: stains
  - **torn_paper**: torn paper
  - **blue**: blueish tone
  - **faded**: faded effect
- **Layout**:
  - **picture_overlay**: overlay of another image in a corner
  - **text_overlay**: text around or over the picture
  - **cadre**: non-rectangular frame shape
  - **patchwork**: parts of other pictures around 
  - **photo_montage**: new elements pasted onto the picture
  - **encadre_rectangle**: rectangular frame around the picture
- **Human made corrections**:
  - **erased_element**: an object/element was erased
  - **drawing**: the picture is a drawing
  - **paint**: the picture is a painting or has traces of paint on it

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
