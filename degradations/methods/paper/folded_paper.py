import cv2
import numpy as np
import random
import glob


# --- Method 1: draw fold lines on the paper ---

def generate_fold_lines(shape, num_folds=3, thickness=2, intensity=60, num_creases=10):
    """
    Generates fold lines with slight waviness and crumples.

    Parameters:
        shape (tuple): Image dimensions (height, width).
        num_folds (int): Number of vertical and horizontal folds.
        thickness (int): Thickness of the fold lines.
        intensity (int): Contrast of the fold lines.

    Returns:
        numpy.ndarray: Fold pattern in grayscale.
    """
    height, width = shape
    fold_pattern = np.ones((height, width), dtype=np.uint8) * 128  # Mid-gray base

    # Define fold positions
    fold_positions_x = np.linspace(0, width, num_folds + 2, dtype=int)[1:-1]
    fold_positions_y = np.linspace(0, height, num_folds + 2, dtype=int)[1:-1]

    # soft lines
    for x in fold_positions_x:
        points = [(x + random.randint(-1, 1), y) for y in range(0, height, 10)]
        points = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(fold_pattern, [points], isClosed=False, color=128 - intensity, thickness=thickness * 2)
        cv2.polylines(fold_pattern, [points], isClosed=False, color=128 + intensity, thickness=thickness)

    for y in fold_positions_y:
        points = [(x, y + random.randint(-1, 1)) for x in range(0, width, 10)]
        points = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(fold_pattern, [points], isClosed=False, color=128 - intensity, thickness=thickness * 2)
        cv2.polylines(fold_pattern, [points], isClosed=False, color=128 + intensity, thickness=thickness)

    # Apply slight Gaussian blur to blend
    fold_pattern = cv2.GaussianBlur(fold_pattern, (5, 5), 3)

    # strong central black lines
    for x in fold_positions_x:
        points = [(x + random.randint(-1, 1), y) for y in range(0, height, 10)]
        points = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(fold_pattern, [points], isClosed=False, color=256, thickness=1)

    for y in fold_positions_y:
        points = [(x, y + random.randint(-1, 1)) for x in range(0, width, 10)]
        points = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(fold_pattern, [points], isClosed=False, color=256, thickness=1)
    """
    # Add random crumples
    for _ in range(num_creases):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = x1 + random.randint(-width//4, width//4), y1 + random.randint(-height//4, height//4)
        points_1 = [(x1 + random.randint(-1, 1), y1) for y1 in range(0, height, 10)]
        points_1 = np.array(points_1, np.int32).reshape((-1, 1, 2))
        points_2 = [(x2,y2 + random.randint(-1, 1)) for x2 in range(0, width, 10)]
        points_2 = np.array(points_2, np.int32).reshape((-1, 1, 2))
        
        cv2.polylines(fold_pattern, [points_1], 128 - intensity, thickness//2)
        cv2.polylines(fold_pattern, [points_2], 128 - intensity, thickness//2)
    """
    fold_pattern = cv2.bitwise_not(fold_pattern)
    
    #noise = np.random.normal(0,15, (height, width)).astype(np.uint8)
    #fold_pattern = cv2.subtract(fold_pattern, noise)
    return fold_pattern


def fold_effect(img, num_folds=2, thickness=2, intensity=80):
    """
    Applies a fold lines effect to image.

    Parameters:
        img (numpy.ndarray): Input image.
        num_folds (int): Number of folds.
        thickness (int): Thickness of fold lines.
        intensity (int): Visibility of fold lines.

    Returns:
        numpy.ndarray: Image with fold marks.
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    folds = generate_fold_lines(gray_img.shape, num_folds, thickness, intensity)
    
    # Blend using addWeighted for a more natural look
    folded_paper = cv2.addWeighted(gray_img, 0.7, folds, 0.3, 0)

    return folded_paper





# --- Method 2: apply a texture from a random crumpled paper image ---

def folded_paper(img, intensity = 0.4):
    """
    Applies a folded paper texture onto an image.
    """
    # grey image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    texture_files = glob.glob("../datasets/folded_texture/*")
    texture_path = np.random.choice(texture_files)    
    texture_path = texture_path.replace("\\", "/")
    texture = cv2.imdecode(np.fromfile(texture_path, np.uint8), cv2.IMREAD_GRAYSCALE)

    # Resize texture to match image size
    texture = cv2.resize(texture, (img.shape[1], img.shape[0]))
 
    # Blend using multiply
    blended = cv2.addWeighted(gray_img,(1-intensity),texture,intensity,0)
    return blended


import torch.nn as nn
class transforms_folded_paper(nn.Module):
    def __init__(self, intensity=10):
        super(transforms_folded_paper, self).__init__()
        self.intensity = intensity

    def __call__(self, batch):
        for image in batch:
            image = folded_paper(image, intensity=self.intensity)
        return batch




if __name__ == "__main__":

    # Example usage
    path = "degradations/datasets/original/"
    img = cv2.imread(path+"FRAN_0568_000183_L.jpg")
    #folded_img = fold_effect(img, num_folds=2, thickness=2, intensity=70)
    folded_img = folded_paper(img)

    cv2.imshow("Folded Paper Effect", folded_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
