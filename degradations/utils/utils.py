import cv2
import os
import numpy as np

def load_image(input_path, to_grayscale=True):
    """
    Loads an image from a file.

    Args:
        input_path (str): Path to the input image.
        to_grayscale (bool): Whether to convert the image to grayscale.

    Returns:
        numpy.ndarray: Loaded image.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE if to_grayscale else cv2.IMREAD_COLOR)
    
    alpha = 0.8 # Contrast control
    beta = 10 # Brightness control

    # call convertScaleAbs function
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    print("image equalized")
    # define the alpha and beta
    return image

def save_image(output_path, image):
    """
    Saves an image to a file.

    Args:
        output_path (str): Path to save the image.
        image (numpy.ndarray): Image to save.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cv2.imwrite(output_path, image)
    print(f"Image saved to: {output_path}")

def resize_image(image, downscale_factor):
    """
    Resizes an image by a given factor.

    Args:
        image (numpy.ndarray): Input image.
        downscale_factor (float): Factor to downscale the image.

    Returns:
        numpy.ndarray: Resized image.
    """
    new_size = (int(image.shape[1] * downscale_factor), int(image.shape[0] * downscale_factor))
    print(f"Image resized to: {new_size}")
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)




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
