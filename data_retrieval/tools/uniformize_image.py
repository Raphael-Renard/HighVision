import sys
path = "."
if path not in sys.path:
    sys.path.insert(0, path)

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

def uniformize_image(image_path, size, path=True):
    if path:
        image = Image.open(image_path)
    else: 
        image = image_path
    image_array = np.asarray(image)

    if len(image_array.shape) == 2:
        image_array = np.stack((image_array,) * 3, axis=-1)
        if image_array.max() > 255:
            image_array = image_array / image_array.max()
            image_array = (image_array * 255).astype(np.uint8)
        image = Image.fromarray(image_array).convert('RGB')
    
    elif image_array.shape[2]==4:
        image_array = np.asarray(image)
        image = Image.fromarray(image_array).convert('RGB')

    n, m, _ = image_array.shape
    max_dim = max(n, m)
    ratio = size / max_dim
    
    new_size = (int(m * ratio), int(n * ratio))

    resized_image = image.resize(new_size)
    resized_array = np.array(resized_image)
    start_x = int((size - resized_array.shape[1]) / 2)
    end_x = start_x + resized_array.shape[1]
    start_y = int((size - resized_array.shape[0]) / 2)
    end_y = start_y + resized_array.shape[0]
    
    final_image = np.zeros((size, size, 3), dtype=np.uint8)
    final_image[start_y:end_y, start_x:end_x] = resized_array
    
    return final_image