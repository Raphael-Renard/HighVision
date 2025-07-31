import cv2
import numpy as np
import torch 
import torch.nn as nn
import glob
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from absolute_path import absolutePath


def photo_montage(img, contour_thickness=1):
    """
    Colle un objet aléatoire entouré d'un contour blanc provenant d'une autre photo sur l'image

    contour_thickness = taille du contour blanc autour de l'objet collé
    """

    image = img.copy()
    height, width, _ = image.shape

    object_files = glob.glob(absolutePath+"degradations/datasets/objets_extraits/*")
    object_path = np.random.choice(object_files)    
    object_path = object_path.replace("\\", "/")
    obj = cv2.imdecode(np.fromfile(object_path, np.uint8), cv2.IMREAD_UNCHANGED)

    # Redimensionner l'objet (facultatif selon besoin)
    min_scale = 1/5
    max_scale = 2/3
    obj_h, obj_w = obj.shape[:2]

    # Calculer les rapports max pour que l'objet rentre dans l'image
    scale_w = (width * np.random.uniform(min_scale, max_scale)) / obj_w
    scale_h = (height * np.random.uniform(min_scale, max_scale)) / obj_h
    scale = min(scale_w, scale_h)

    # Redimensionner l'objet
    new_size = (int(obj_w * scale), int(obj_h * scale))
    obj = cv2.resize(obj, new_size, interpolation=cv2.INTER_AREA)
    if len(obj.shape) == 2:
        obj = cv2.cvtColor(obj, cv2.COLOR_GRAY2BGRA)
    elif obj.shape[2] == 3:
        obj = cv2.cvtColor(obj, cv2.COLOR_BGR2BGRA)

    # Séparer les canaux BGR et alpha
    obj_bgr = obj[:, :, :3]
    alpha = obj[:, :, 3]

    # Créer contour blanc
    kernel = np.ones((contour_thickness*2+1, contour_thickness*2+1), np.uint8)
    dilated_alpha = cv2.dilate(alpha, kernel)
    contour_mask = dilated_alpha - alpha

    white_contour = np.zeros_like(obj_bgr)
    white_contour[:, :] = (255, 255, 255)

    # Appliquer le contour
    x_offset = np.random.randint(0, width - obj.shape[1])
    y_offset = np.random.randint(0, height - obj.shape[0])
    roi = np.zeros_like(image[y_offset:y_offset+obj.shape[0], x_offset:x_offset+obj.shape[1]])
    for c in range(3):
        roi[:,:,c] = image[y_offset:y_offset+obj.shape[0], x_offset:x_offset+obj.shape[1], c]
        roi[:,:,c] = np.where(contour_mask > 0, white_contour[:, :, c], roi[:,:,c])
        roi[:,:,c] = np.where(alpha > 0, obj_bgr[:, :, c], roi[:,:,c])

    image[y_offset:y_offset+obj.shape[0], x_offset:x_offset+obj.shape[1]] = roi

    return image


class transforms_photo_montage(nn.Module):
    def __init__(self, contour_thickness=1):
        super(transforms_photo_montage, self).__init__()
        self.contour_thickness = contour_thickness

    def __call__(self, batch):
        one_image = False
        if len(batch.shape) == 3: # si on ne passe qu'une image au lieu d'un batch
            batch = batch.unsqueeze(0)
            one_image = True
            
        results = torch.empty_like(batch)
        for i, image in enumerate(batch):
            image_array = np.transpose(np.array(image), (1, 2, 0)).copy() * 255
            image_array = image_array.astype(np.uint8)
            image = photo_montage(image_array, self.contour_thickness)
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image)
            results[i] = image/255
        
        if one_image:
            results = results.squeeze(0)
        return results

