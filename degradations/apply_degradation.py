from utils.utils import load_image, save_image, resize_image
from methods.halftoning.floyd_steinberg import floyd_steinberg_halftoning
from methods.halftoning.atkinson import atkinson_dithering
from methods.halftoning.bayers_threshold import bayer_halftoning
from methods.halftoning.dot_traditional import dot_halftoning
from methods.noise.gaussian_noise import add_gaussian_noise
from methods.noise.salt_and_pepper import add_salt_and_pepper_noise
from methods.noise.film_grain import apply_film_grain
from methods.noise.dirty_rollers import dirty_rollers
from methods.paper.ink_bleed import ink_bleed
from methods.paper.crumpled_paper import crumpled_paper
from methods.paper.folded_paper import folded_paper
from methods.paper.folded_paper import fold_effect
from methods.paper.bleedthrough import bleedthrough
from methods.paper.bleedthrough import bleedthrough_other
from methods.paper.scribbles import scribbles
from methods.paper.stains import stains
from methods.human_corrections.erased_element import erased_element
from methods.layout.picture_overlay import picture_overlay
from methods.layout.text_overlay import text_overlay



import os
import cv2

def apply_degradation(method, input_path, output_folder, downscale_factor=1, **kwargs):
    """
    Applies a specified degradation method to an image and saves the result.

    Args:
        method (function): The degradation method to apply.
        input_path (str): Path to the input image.
        output_folder (str): Folder to save the output image.
        downscale_factor (float): Factor to downscale the image before applying the method.
        **kwargs: Additional parameters for the degradation method.

    Returns:
        str: Path to the saved output image.
    """
    # Load the image
    image = load_image(input_path, to_grayscale=True)
    image = cv2.equalizeHist(image)
    og_size = image.shape
    # Resize the image if necessary
    print(downscale_factor)
    
    if downscale_factor < 1.0:
        image = resize_image(image, downscale_factor)

    # Apply the degradation method
    degraded_image = method(image, **kwargs)

    # Generate output path
    method_name = method.__name__
    image_name = os.path.basename(input_path).split('.')[0]
    output_path = os.path.join(output_folder, f"{image_name}_{method_name}.jpg")

    # Save the degraded image
    #resizing to original size
    if downscale_factor<1.0:
        degraded_image = cv2.resize(degraded_image, (og_size[1], og_size[0]), interpolation=cv2.INTER_NEAREST)
    save_image(output_path, degraded_image)
    return output_path

def main():
    # Example usage of the library
    input_path = "../datasets/original/FRAN_0568_000220_L.jpg"
    output_folder = "../datasets/degraded/"
    downscale_factor = 0.7 #optional
    print("Applying degradation...")
    apply_degradation(
        method=floyd_steinberg_halftoning,
        input_path=input_path,
        output_folder=output_folder,
        downscale_factor=0.5,
        block_size=8# Parameter specific to the method
        
    )

if __name__ == "__main__":
    main()
