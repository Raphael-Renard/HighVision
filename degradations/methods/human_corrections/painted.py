import cv2
import numpy as np
import random

def painted(img, num_painted_regions=3, brush_size=25):
    """
    Simulates a human-painted effect by overlaying brush strokes on an image.
    
    Parameters:
        image_path (str): Path to the input image.
        output_path (str): Path to save the modified image.
        num_painted_regions (int): Number of areas to paint over.
        brush_size (int): Size of the simulated brush strokes.
    """
    height, width = img.shape[:2]

    # Convert to grayscale & detect edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray,alpha=1.5,beta=-50) # enhance contrasts
    
    gray = cv2.GaussianBlur(gray, (9, 9), 0) # smooth before edges detection
    edges = cv2.Canny(gray, 50, 150)  # Edge detection to find high-contrast areas

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    painted_img = gray.copy()
    
    # "paint over" contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Extract region & apply "paint effect" (blur)
        region = painted_img[y:y+h, x:x+w]
        blurred = cv2.GaussianBlur(region, (brush_size, brush_size), 0)

        # Replace the region in the original image
        painted_img[y:y+h, x:x+w] = blurred

    return painted_img


if __name__ == "__main__":
    path = "degradations/datasets/original/"
    img = cv2.imread(path+"FRAN_0568_000020_L.jpg")
    img = painted(img)

    
    # resize image for better visualization
    scale_percent = 8 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow("painted", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()