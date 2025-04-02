import cv2
import numpy as np
import random
from ultralytics import YOLO
import torch 
import torch.nn as nn


def photo_montage(img, contour_thickness=1):
    """
    Détecte un objet dans l'image et le copie colle en plus grand ailleurs sur la photo

    contour_thickness = taille du contour blanc autour de l'objet collé
    """
    model = YOLO("yolov8n-seg.pt",verbose=False)
    
    image = img.copy()
    height, width, _ = image.shape
    
    # Détecter les objets
    results = model(image,verbose=False)
    
    masks = results[0].masks.xy  # Liste des contours des objets
    
    if not masks:
        print("Aucun objet détecté.")
        return image
    
    # Choisir un objet au hasard
    obj_mask = random.choice(masks)
    obj_mask = np.array(obj_mask, dtype=np.int32)
    
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [obj_mask], 255)
    
    # Extraire l'objet
    object_crop = cv2.bitwise_and(image, image, mask=mask)
    x, y, w, h = cv2.boundingRect(obj_mask)
    object_crop = object_crop[y:y+h, x:x+w]
    mask_crop = mask[y:y+h, x:x+w]

    
    contour_img = np.zeros_like(object_crop, dtype=np.uint8)
    contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_img, contours, -1, (255, 255, 255), contour_thickness)

    # Ajoute un contour blanc à l'objet détouré
    object_crop = cv2.addWeighted(object_crop, 1, contour_img, 1, 0)
    
    # Agrandit l'objet
    scale_factor = random.uniform(1.5, 2.0) 
    new_width = int(w * scale_factor)
    new_height = int(h * scale_factor)
    resized_object = cv2.resize(object_crop, (new_width, new_height))
    resized_mask = cv2.resize(mask_crop, (new_width, new_height))
    
    # Trouver un emplacement aléatoire
    max_x = width - new_width
    max_y = height - new_height
    if max_x <= 0 or max_y <= 0: # L'objet agrandi est trop grand pour être replacé dans l'image
        resized_object = object_crop
        resized_mask = mask_crop

    new_x = random.randint(0, max_x)
    new_y = random.randint(0, max_y)
    
    # Coller l'objet agrandi
    for c in range(3):  # chaque canal de couleur
        image[new_y:new_y+new_height, new_x:new_x+new_width, c] = np.where(
            resized_mask > 0, resized_object[:, :, c], image[new_y:new_y+new_height, new_x:new_x+new_width, c]
        )
    
    return image


class transforms_photo_montage(nn.Module):
    def __init__(self, contour_thickness=1):
        super(transforms_photo_montage, self).__init__()
        self.contour_thickness = contour_thickness

    def __call__(self, batch):
        results = torch.empty_like(batch)
        for i, image in enumerate(batch):
            image_array = np.transpose(np.array(image), (1, 2, 0)).copy() * 255
            image_array = image_array.astype(np.uint8)
            image = photo_montage(image_array, self.contour_thickness)
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image)
            results[i] = image/255
        return results



if __name__ == "__name__":
    #image_path = "C:/Users/rapha/Documents/Cours/Master/Stage/HighVision/degradations/results/2K2476_16_01.jpg"
    image_path = "C:/Users/rapha/Documents/Cours/Master/Stage/Data/Sena/FRAN_0568_11AR_699/FRAN_0568_000014_L.jpg"
    img = cv2.imread(image_path)
    modified_img = photo_montage(img,30)
    cv2.imwrite("photo_montage_big.jpg", modified_img)
