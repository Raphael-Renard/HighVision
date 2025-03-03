import subprocess
import os

def preprocess(input_path):
    return

def apply_film_grain(input_image_path, output_image_path, gray=True, grain_type=3, power="0.8,0.2,0.1", scale=0, sharpen=0):
    """
    Applies a film grain effect to an image using the filmgrainer library https://github.com/larspontoppidan/filmgrainer via the command line.

    Args:
        input_image_path (str): Path to the input image.
        output_image_path (str): Path to save the output image.
        gray (bool): Whether to apply the effect in grayscale mode.
        grain_type (int): Type of grain (1, 2, or 3).
        power (str): Strength of the grain (comma-separated values).

    Returns:
        None
    """
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input image not found: {input_image_path}")

    # Build the command
    command = [
        "filmgrainer",
        "--type", str(grain_type),
        "--power", power, "--scale",scale, "--sharpen", sharpen,
        "-o", output_image_path,
        input_image_path
    ]

    if gray:
        command.insert(1, "--gray")

    # Execute the command
    try:
        subprocess.run(command, check=True)
        print(f"Film grain effect applied and saved to: {output_image_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error applying film grain: {e}")

    return output_image_path
#inline testing
if __name__ == "__main__":
    input_path = "your input path"
    output_path = "output path"

    # Apply film grain
    apply_film_grain(input_path, output_path, gray=True, grain_type=3, power="0.8,0.2,0.1", scale="3", sharpen="1")
