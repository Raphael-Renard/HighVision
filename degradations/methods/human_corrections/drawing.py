import cv2
import torch
import numpy as np
import torch.nn as nn
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from absolute_path import absolutePath




def cartoon(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray_image.shape
    size_factor = max(height, width) / 500

    # Apply median blur to reduce noise
    blur_ksize = int(5 * size_factor)
    if blur_ksize % 2 == 0: blur_ksize += 1 
    gray_blur = cv2.medianBlur(gray_image, blur_ksize)

    # Détection des contours avec un seuil adaptatif
    block_size = int(9 * size_factor)
    if block_size % 2 == 0: block_size += 1
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, block_size, 9)

    # Appliquer un filtre bilatéral pour lisser l'image
    d = int(9 * size_factor) 
    sigma_color = int(100 * size_factor)  # Influence des couleurs voisines
    sigma_space = int(100 * size_factor)  # Influence de l’espace
    color_image = cv2.bilateralFilter(img, d, sigma_color, sigma_space)

    cartoon = cv2.bitwise_and(color_image, color_image, mask=edges)
    return cartoon







def hed(image, with_aplat_noirs = True, with_hachures=False, flou=5):
    """
    Apply the HED algorithm to an image to obtain its edges.
    The function also allows for the addition of hatching (lines) on dark areas of the image.
    Parameters:
    - image: Input image (numpy array).
    - with_hachures: Boolean indicating whether to add hatching.
    - with_aplat_noirs: Boolean indicating whether to add black areas.
    - flou: Intensity of the blur applied to the black areas.
    """
    

    # Load the pre-trained HED model and its configuration file
    base_path = absolutePath + "degradations/methods/human_corrections/"
    model_path = base_path + "hed_pretrained_bsds.caffemodel"
    config_path = base_path + "deploy.prototxt"

    # Load the model
    net = cv2.dnn.readNetFromCaffe(config_path, model_path)

    # Prepare the image for the model
    img_shape = image.shape
    
    border_size = 35
    crop_size = 70

    image_resized = cv2.resize(image, (500-border_size*2, 500-border_size*2))
    image_resized = cv2.copyMakeBorder(image_resized, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    blob = cv2.dnn.blobFromImage(image_resized, 1.0, (500, 500), (104.0, 177.0, 123.0), swapRB=True, crop=False)

    # Set the input to the model
    net.setInput(blob)
    
    # Run forward pass to get the edge map
    hed_output = net.forward()
    hed_output = hed_output.squeeze()
    
    hed_output = cv2.normalize(hed_output, None, 0, 255, cv2.NORM_MINMAX)
    hed_output = np.uint8(hed_output)
    hed_output = hed_output[crop_size:,crop_size:]
    hed_output = cv2.resize(hed_output, (img_shape[1], img_shape[0]))


    # Shading

    # Ajouter des hachures sur les zones sombres de l'image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    seuil_noir = 50       # Aplat noir sur les parties très foncées
    seuil_gris_fonce = 100 # Hachures sur les parties gris foncé

    # Créer les masques
    masque_noir = gray < seuil_noir
    masque_gris = (gray >= seuil_noir) & (gray < seuil_gris_fonce)
    combined = hed_output.copy()

    if with_aplat_noirs:
        # Créer les aplats noirs
        aplats_noirs = np.zeros_like(hed_output)
        aplats_noirs[masque_noir] = 255 
        aplats_noirs = cv2.blur(aplats_noirs, (flou, flou))
        
        combined = cv2.addWeighted(combined, 1, aplats_noirs, 1, 0)

    if with_hachures:
        # Créer les hachures
        hachures = np.zeros_like(hed_output)

        line_thickness = max(1, int(img_shape[0] / 200))
        line_spacing = max(5, int(img_shape[0] / 40))

        for i in range(0, img_shape[0], line_spacing):
            cv2.line(hachures, (0, i), (img_shape[1], i + img_shape[1]), 255, line_thickness)

        # Appliquer le masque pour n’avoir les hachures que sur les zones gris foncé
        hachures_mask = np.zeros_like(hed_output)
        hachures_mask[masque_gris] = hachures[masque_gris]

        combined = cv2.addWeighted(combined, 1, hachures_mask, 1, 0)

    if not with_aplat_noirs and not with_hachures:
        combined = hed_output
 
    final_output = cv2.bitwise_not(combined)
    final_output = cv2.cvtColor(final_output, cv2.COLOR_GRAY2BGR)
    return final_output

 

class transforms_drawing(nn.Module):
    def __init__(self, aplat_noirs=None, hachures=None, flou=5):
        super(transforms_drawing, self).__init__()
        self.aplat_noirs = aplat_noirs
        self.hachures = hachures
        self.flou = flou

    def __call__(self, batch):
        one_image = False
        if len(batch.shape) == 3: # si on ne passe qu'une image au lieu d'un batch
            batch = batch.unsqueeze(0)
            one_image = True
            
        results = torch.empty_like(batch)
        for i, image in enumerate(batch):
            image_array = np.transpose(np.array(image), (1, 2, 0)).copy() * 255
            image_array = image_array.astype(np.uint8)

            if self.aplat_noirs is None:
                aplat_noirs = random.choice([True, False])
            else:
                aplat_noirs = self.aplat_noirs
            
            if self.hachures is None:
                hachures = random.choice([True, False])
            else:
                hachures = self.hachures
            

            image = hed(image_array, aplat_noirs, hachures, self.flou)
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image)
            results[i] = image/255
        
        if one_image:
            results = results.squeeze(0)
        return results



