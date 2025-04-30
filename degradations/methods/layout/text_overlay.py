import cv2
import numpy as np
import random
import string
import torch
import torch.nn as nn


def remove_black_borders(image):
    """Supprime les bords noirs qui ont été rajoutés pour le dataloader."""
    # Créer un masque détectant les pixels non noirs
    mask = np.any(image > 0, axis=2)
    
    # Trouver les coordonnées des pixels non noirs
    coords = np.column_stack(np.where(mask))

    if len(coords) == 0:
        return image

    # Trouver la boîte englobante des pixels non noirs
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Recadre image
    cropped_image = image[y_min:y_max+1, x_min:x_max+1]

    return cropped_image, (y_min, x_min, y_max, x_max)  # image recadrée et coordonnées du recadrage

def restore_black_borders(original_shape, cropped_image, crop_coords):
    """Restaure les bords noirs après traitement, en gardant la taille originale."""
    h_original, w_original = original_shape[:2]
    h_cropped, w_cropped = cropped_image.shape[:2]

    # Créer une image noire avec la taille originale
    restored_image = np.zeros((h_original, w_original, 3), dtype=np.uint8)

    # Récupérer les anciennes coordonnées de la zone non noire
    y_min, x_min, _, _ = crop_coords

    # Coller l’image recadrée au bon emplacement
    restored_image[y_min:y_min+h_cropped, x_min:x_min+w_cropped] = cropped_image

    return restored_image



def text_overlay(img, font_size_1=None):
    """
    Add two lines of random black text on a white rectangle in one corner or at the top of the image
    font_size_1 (int): size of the first line of text
    length_txt1 (int): number of characters in the first line of the text
    length_txt2 (int): number of characters in the second line of the text
    """

    # pick a corner of the image at random (where the text will be)
    corner = np.random.randint(0, 5)

    # corner that will be overlayed by text
    if corner==4: # text at the top
        ratio_height, ratio_width = np.random.uniform(0.4, 0.6),np.random.uniform(0.2, 0.3)
    else: # in one corner
        ratio_height, ratio_width = np.random.uniform(0.4, 0.7),np.random.uniform(0.7, 0.9)
    half_height, half_width = img.shape[0] // 2, img.shape[1] // 2
    corner_height, corner_width = int(half_height*ratio_height), int(half_width*ratio_width)
    border = img.shape[0]//30
    
    # text
    if font_size_1 is None:
        font_size_1 = corner_height//70
        thickness_1 = 4*font_size_1
        font_size_2 = font_size_1//2
        thickness_2 = 6*font_size_2

        if corner==4: # top center
            length_txt1 = (img.shape[1]-2*corner_width)//320
            length_txt2 = (img.shape[1]-2*corner_width)//200
        else: # corner
            length_txt1 = corner_width//320
            length_txt2 = corner_width//200
    else:
        if corner==4: # top center
            length_txt1 = (img.shape[1]-2*corner_width)//20
            length_txt2 = (img.shape[1]-2*corner_width)//20
        else: # corner
            length_txt1 = corner_width//20
            length_txt2 = corner_width//20

        thickness_1 = 3*font_size_1
        font_size_2 = font_size_1//2
        thickness_2 = 4*font_size_2

    if font_size_2==0:
        font_size_2 = 1
        thickness_2 = 1
    
    
    text1 = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase) for _ in range(length_txt1)).strip()
    text2 = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase) for _ in range(length_txt2)).strip()

    if corner == 0: # top left corner
        img[0:corner_height, 0:corner_width] = 255

        # write random text in the white rectangle
        cv2.putText(img,text1,(0,corner_height//3 + border),
                    cv2.FONT_HERSHEY_COMPLEX,font_size_1,(0,0,0),thickness_1,cv2.LINE_AA)
        cv2.putText(img,text2,(0,2*corner_height//3 + border),
                    cv2.FONT_HERSHEY_COMPLEX,font_size_2,(0,0,0),thickness_2,cv2.LINE_AA)
        
    elif corner == 1: # top right corner
        img[0:corner_height, img.shape[1] - corner_width:img.shape[1]] = 255

        # write random text in the white rectangle
        cv2.putText(img,text1,(img.shape[1] - corner_width + border, corner_height//3 + border),
                    cv2.FONT_HERSHEY_COMPLEX,font_size_1,(0,0,0),thickness_1,cv2.LINE_AA)
        cv2.putText(img,text2,(img.shape[1] - corner_width + border*2, 2*corner_height//3 + 2*border),
                    cv2.FONT_HERSHEY_COMPLEX,font_size_2,(0,0,0),thickness_2,cv2.LINE_AA)

     
    elif corner == 2: # bottom right corner
        img[img.shape[0] - corner_height:img.shape[0], img.shape[1] - corner_width:img.shape[1]] = 255

        # write random text in the white rectangle
        cv2.putText(img,text1,(img.shape[1] - corner_width + border, img.shape[0] - corner_height + corner_height//3 + border),
                    cv2.FONT_HERSHEY_COMPLEX,font_size_1,(0,0,0),thickness_1,cv2.LINE_AA)
        cv2.putText(img,text2,(img.shape[1] - corner_width + border*2, img.shape[0] - corner_height + 2*corner_height//3 + 2*border),
                    cv2.FONT_HERSHEY_COMPLEX,font_size_2,(0,0,0),thickness_2,cv2.LINE_AA)


    elif corner == 3: # bottom left corner
        img[img.shape[0] - corner_height:img.shape[0], 0:corner_width] = 255

        # write random text in the white rectangle
        cv2.putText(img,text1,(0,img.shape[0] - corner_height + corner_height//3 + border),
                    cv2.FONT_HERSHEY_COMPLEX,font_size_1,(0,0,0),thickness_1,cv2.LINE_AA)
        cv2.putText(img,text2,(0,img.shape[0] - corner_height+ 2*corner_height//3 + 2*border),
                    cv2.FONT_HERSHEY_COMPLEX,font_size_2,(0,0,0),thickness_2,cv2.LINE_AA)
    
    else: # center
        img[0:corner_height, corner_width : img.shape[1] - corner_width] = 255

        # write random text in the white rectangle
        cv2.putText(img,text1,(corner_width + border, corner_height//3+ border),
                    cv2.FONT_HERSHEY_COMPLEX,font_size_1,(0,0,0),thickness_1,cv2.LINE_AA)
        cv2.putText(img,text2,(corner_width + border*2, 2*corner_height//3 + 2*border),
                    cv2.FONT_HERSHEY_COMPLEX,font_size_2,(0,0,0),thickness_2,cv2.LINE_AA)

    return img




class transforms_text_overlay(nn.Module):
    def __init__(self, font_size_1=1):
        super(transforms_text_overlay, self).__init__()
        self.font_size_1 = font_size_1

    def __call__(self, batch):
        one_image = False
        if len(batch.shape) == 3: # si on ne passe qu'une image au lieu d'un batch
            batch = batch.unsqueeze(0)
            one_image = True
            
        results = torch.empty_like(batch)
        for i, image in enumerate(batch):
            image_array = np.transpose(np.array(image), (1, 2, 0)).copy() * 255
            image_array = image_array.astype(np.uint8)
            shape = image_array.shape
            image, black_borders = remove_black_borders(image_array)

            image = text_overlay(image, self.font_size_1) 

            image = restore_black_borders(shape, image, black_borders)
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image) / 255
            results[i] = image
        
        if one_image:
            results = results.squeeze(0)
        return results
    


if __name__ =="__main__":
    image_path = "C:/Users/rapha/Documents/Cours/Master/Stage/HighVision/degradations/results/2K2476_16_01.jpg"
    image_path = "C:/Users/rapha/Documents/Cours/Master/Stage/Data/Sena/FRAN_0568_11AR_699/FRAN_0568_000014_L.jpg"
    img = cv2.imread(image_path)
    shape = img.shape
    #img, black_borders = remove_black_borders(img)
    img = text_overlay(img,None)
    #img = restore_black_borders(shape, img, black_borders)
    cv2.imwrite("text_overlay.jpg",img)