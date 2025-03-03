import cv2
import os

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
