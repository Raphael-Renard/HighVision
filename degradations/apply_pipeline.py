import os
from apply_degradation import apply_degradation
from utils.utils import load_image, save_image
from methods.halftoning.floyd_steinberg import floyd_steinberg_halftoning
from methods.halftoning.atkinson import atkinson_dithering
from methods.halftoning.bayers_threshold import bayer_halftoning
from methods.halftoning.dot_traditional import dot_halftoning  # Import your halftoning methods
from methods.noise.gaussian_noise import add_gaussian_noise
from methods.noise.salt_and_pepper import add_salt_and_pepper_noise
from methods.noise.dirty_rollers import dirty_rollers
from methods.noise.film_grain import  apply_film_grain # Import your noise methods
from methods.paper.ink_bleed import ink_bleed  
from methods.paper.crumpled_paper import crumpled_paper
from methods.paper.folded_paper import folded_paper
from methods.paper.folded_paper import fold_effect
from methods.paper.bleedthrough import bleedthrough
from methods.paper.bleedthrough import bleedthrough_other
from methods.paper.scribbles import scribbles
from methods.paper.stains import stains # Import your paper feel methods
from methods.human_corrections.erased_element import erased_element # Import your human correction methods
from methods.layout.picture_overlay import picture_overlay
from methods.layout.text_overlay import text_overlay # Import your layout methods


def apply_pipeline(input_path, output_folder, halftoning_method, noise_method, paper_method,
                   halftoning_args=None, noise_args=None, paper_args=None, downscale_factor_halftone=1.0, downscale_factor_noise=1.0, downscale_factor_paper=1.0):
    """
    Applies a pipeline of degradations to an image: halftoning, noise, and paper feel.

    Args:
        input_path (str): Path to the input image.
        output_folder (str): Folder to save the output image.
        halftoning_method (function): Halftoning method to apply.
        noise_method (function): Noise method to apply.
        paper_method (function): Paper feel method to apply.
        halftoning_args (dict, optional): Arguments for the halftoning method.
        noise_args (dict, optional): Arguments for the noise method.
        paper_args (dict, optional): Arguments for the paper feel method.
        downscale_factor (float, optional): Downscaling factor for the image.

    Returns:
        str: Path to the final saved image.
    """
    halftoning_args = halftoning_args or {}
    noise_args = noise_args or {}
    paper_args = paper_args or {}

    # Temporary folder to store intermediate results
    temp_folder = os.path.join(output_folder, "temp")
    os.makedirs(temp_folder, exist_ok=True)

    # Step 1: Apply Halftoning
    halftone_output_path = apply_degradation(halftoning_method, input_path, temp_folder, downscale_factor_halftone, **halftoning_args)

    # Step 2: Apply Noise
    noise_input_path = halftone_output_path
    noise_output_path = apply_degradation(noise_method, noise_input_path, temp_folder, downscale_factor_noise, **noise_args)

    # Step 3: Apply Paper Feel
    paper_input_path = noise_output_path
    final_output_path = apply_degradation(paper_method, paper_input_path, output_folder, downscale_factor_paper, **paper_args)

    # Clean up intermediate results (optional)
    for file in os.listdir(temp_folder):
        os.remove(os.path.join(temp_folder, file))
    os.rmdir(temp_folder)

    return final_output_path


# Inline Testing
if __name__ == "__main__":
    input_image_path = "../datasets/original/FRAN_0568_000220_L.jpg"
    output_folder = "../datasets/degraded/pipeline"
    
    # Example pipeline
    final_image_path = apply_pipeline(
        input_path=input_image_path,
        output_folder=output_folder,
        halftoning_method=floyd_steinberg_halftoning,
        noise_method=add_gaussian_noise,
        paper_method=ink_bleed,
        halftoning_args={"block_size":16},
        noise_args={"mean":0, "stddev":10},
        paper_args={},
        downscale_factor_halftone=0.7,
        downscale_factor_noise=0.1,
        downscale_factor_paper=0.7
    )
    print(f"Pipeline complete. Final image saved at: {final_image_path}")
