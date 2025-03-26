import cv2
import numpy as np
import torch.nn as nn
import torch



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



def picture_overlay(img, img2=None, thickness=None):
    """
    Add part of another picture in the corner of the image
    thickness (int): the two images will be separated by a white line of this thickness
    """

    if img2 is None:
        img2 = img.copy()

    img_copy = img.copy()

    # corner of the second image to overlay
    ratio_height, ratio_width = np.random.uniform(0.4, 0.5),np.random.uniform(0.4, 0.5) # part of the corner of the second picture to overlay
    half_height, half_width = img2.shape[0] // 2, img2.shape[1] // 2
    corner_height, corner_width = int(half_height*ratio_height), int(half_width*ratio_width)

    # pick a corner of the first image at random (where the second image will be overlayed)
    corner = np.random.randint(0, 4)

    if thickness is None:
        thickness = max(img.shape[0], img.shape[1]) // 150


    cut_corner = np.random.randint(0, 2) # is the corner of the second image cut or not
 
    if cut_corner == 1:
        cut_height, cut_width = int(corner_height*np.random.uniform(0.3, 0.5)), int(corner_width*np.random.uniform(0.3, 0.5))


    # overlay a corner of the second image on the first image
    

    if corner == 0: # top left corner
        img[0:corner_height, 0:corner_width] = img2[-corner_height:,-corner_width:]

        if cut_corner == 1:
            img[cut_height:corner_height, cut_width:corner_width] = img_copy[cut_height:corner_height, cut_width:corner_width]
            
            # add a white line to separate the two images
            cv2.line(img, (cut_width, cut_height), (corner_width, cut_height), (255, 255, 255), thickness)
            cv2.line(img, (cut_width, cut_height), (cut_width, corner_height), (255, 255, 255), thickness)
            cv2.line(img, (cut_width, corner_height), (0, corner_height), (255, 255, 255), thickness)
            cv2.line(img, (corner_width, cut_height), (corner_width, 0), (255, 255, 255), thickness)
        
        else:
            # add a white line to separate the two images
            cv2.line(img, (0, corner_height), (corner_width, corner_height), (255, 255, 255), thickness)
            cv2.line(img, (corner_width, 0), (corner_width, corner_height), (255, 255, 255), thickness)


    elif corner == 1: # top right corner
        img[0:corner_height, img.shape[1] - corner_width:img.shape[1]] = img2[-corner_height:,:corner_width]

        if cut_corner == 1:
            img[cut_height:corner_height, img.shape[1] - corner_width:img.shape[1] - cut_width] = img_copy[cut_height:corner_height, img.shape[1] - corner_width:img.shape[1] - cut_width]
            
            # add a white line to separate the two images
            cv2.line(img, (img.shape[1] - cut_width, cut_height), (img.shape[1] - corner_width, cut_height), (255, 255, 255), thickness)
            cv2.line(img, (img.shape[1] - cut_width, cut_height), (img.shape[1] - cut_width, corner_height), (255, 255, 255), thickness)
            cv2.line(img, (img.shape[1] - cut_width, corner_height), (img.shape[1], corner_height), (255, 255, 255), thickness)
            cv2.line(img, (img.shape[1] - corner_width, cut_height), (img.shape[1] - corner_width, 0), (255, 255, 255), thickness)

        else:
            # add a white line to separate the two images
            cv2.line(img, (img.shape[1], corner_height), (img.shape[1] - corner_width, corner_height), (255, 255, 255), thickness)
            cv2.line(img, (img.shape[1] - corner_width, 0), (img.shape[1] - corner_width, corner_height), (255, 255, 255), thickness)


    elif corner == 2: # bottom right corner
        img[img.shape[0] - corner_height:img.shape[0], img.shape[1] - corner_width:img.shape[1]] = img2[:corner_height,:corner_width]

        if cut_corner == 1:
            img[img.shape[0] - corner_height:img.shape[0] - cut_height, img.shape[1] - corner_width:img.shape[1] - cut_width] = img_copy[img.shape[0] - corner_height:img.shape[0] - cut_height, img.shape[1] - corner_width:img.shape[1] - cut_width]

            # add a white line to separate the two images
            cv2.line(img, (img.shape[1] - cut_width, img.shape[0] - cut_height), (img.shape[1] - corner_width, img.shape[0] - cut_height), (255, 255, 255), thickness)
            cv2.line(img, (img.shape[1] - cut_width, img.shape[0] - cut_height), (img.shape[1] - cut_width, img.shape[0] - corner_height), (255, 255, 255), thickness)
            cv2.line(img, (img.shape[1] - cut_width, img.shape[0] - corner_height), (img.shape[1], img.shape[0] - corner_height), (255, 255, 255), thickness)
            cv2.line(img, (img.shape[1] - corner_width, img.shape[0] - cut_height), (img.shape[1] - corner_width, img.shape[0]), (255, 255, 255), thickness)

        else:
            # add a white line to separate the two images
            cv2.line(img, (img.shape[1], img.shape[0] - corner_height), (img.shape[1] - corner_width, img.shape[0] - corner_height), (255, 255, 255), thickness)
            cv2.line(img, (img.shape[1] - corner_width, img.shape[0]), (img.shape[1] - corner_width, img.shape[0] - corner_height), (255, 255, 255), thickness)


    else: # bottom left corner
        img[img.shape[0] - corner_height:img.shape[0], 0:corner_width] = img2[:corner_height,-corner_width:]

        if cut_corner == 1:
            img[img.shape[0] - corner_height:img.shape[0] - cut_height, cut_width:corner_width] = img_copy[img.shape[0] - corner_height:img.shape[0] - cut_height, cut_width:corner_width]

            # add a white line to separate the two images
            cv2.line(img, (cut_width, img.shape[0] - cut_height), (corner_width, img.shape[0] - cut_height), (255, 255, 255), thickness)
            cv2.line(img, (cut_width, img.shape[0] - cut_height), (cut_width, img.shape[0] - corner_height), (255, 255, 255), thickness)
            cv2.line(img, (cut_width, img.shape[0] - corner_height), (0, img.shape[0] - corner_height), (255, 255, 255), thickness)
            cv2.line(img, (corner_width, img.shape[0] - cut_height), (corner_width, img.shape[0]), (255, 255, 255), thickness)

        else:
            # add a white line to separate the two images
            cv2.line(img, (0, img.shape[0] - corner_height), (corner_width, img.shape[0] - corner_height), (255, 255, 255), thickness)
            cv2.line(img, (corner_width, img.shape[0]), (corner_width, img.shape[0] - corner_height), (255, 255, 255), thickness)

    return img




class transforms_picture_overlay(nn.Module):
    def __init__(self, thickness=3):
        super(transforms_picture_overlay, self).__init__()
        self.thickness = thickness

    def __call__(self, batch):
        results = torch.empty_like(batch)
        
        for i, image in enumerate(batch):
            image_array = np.transpose(np.array(image), (1, 2, 0)).copy() * 255
            image_array = image_array.astype(np.uint8)
            shape = image_array.shape
            image, black_borders = remove_black_borders(image_array)

            image = picture_overlay(image, thickness = self.thickness)
            
            image = restore_black_borders(shape, image, black_borders)
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image)
            results[i] = image / 255
        return results



if __name__ =="__main__":
    image_path = "C:/Users/rapha/Documents/Cours/Master/Stage/HighVision/degradations/results/2K2476_16_01.jpg"
    #image_path = "C:/Users/rapha/Documents/Cours/Master/Stage/Data/Sena/FRAN_0568_11AR_699/FRAN_0568_000014_L.jpg"
    img = cv2.imread(image_path)
    shape = img.shape
    img, black_borders = remove_black_borders(img)

    img = picture_overlay(img)
    img = restore_black_borders(shape, img, black_borders)
    cv2.imwrite("picture_overlay.jpg",img)