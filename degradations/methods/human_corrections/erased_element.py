import cv2
from ultralytics import YOLO
import numpy as np

def erased_element(img, blur_intensity=101, verbose=False):
    """Detects and removes the largest object outside the center region using YOLO."""

    # Load YOLOv8 model
    model = YOLO("yolov8n.pt", verbose=False)
    
    # Define center region
    height, width = img.shape[:2]
    center_region = (width * 0.25, height * 0.25, width * 0.75, height * 0.75)

    # Run YOLO detection
    results = model(img,verbose=False)

    # Create a mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Filter objects: large & outside center
    large_outer_objects = []
    for result in results:
        for box in result.boxes.xyxy:  # Bounding boxes
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)

            # Check if object is outside the center region
            if (x2 < center_region[0] or x1 > center_region[2] or 
                y2 < center_region[1] or y1 > center_region[3]):
                large_outer_objects.append((x1, y1, x2, y2, area))

    # Sort by largest area first
    large_outer_objects.sort(key=lambda obj: obj[4], reverse=True)

    if not large_outer_objects:
        if verbose:
            print("No large outer objects found.")
        return img

    # Remove the largest detected object
    x1, y1, x2, y2, _ = large_outer_objects[0]

    # draw the detected object as white on the mask
    mask[y1:y2, x1:x2] = 255

    # Inpaint the object to erase it
    inpainted_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    # add a blur effect to the inpainted region
    inpainted_img[y1:y2, x1:x2] = cv2.GaussianBlur(inpainted_img[y1:y2, x1:x2], (blur_intensity, blur_intensity), 0)

    return inpainted_img





import torch.nn as nn
import torch
class transforms_erased_element(nn.Module):
    def __init__(self, blur_intensity=101):
        super(transforms_erased_element, self).__init__()
        self.blur_intensity = blur_intensity

    def __call__(self, batch):
        results = torch.empty_like(batch)
        for i, image in enumerate(batch):
            image_array = np.transpose(np.array(image), (1, 2, 0)).copy() * 255
            image_array = image_array.astype(np.uint8)
            image = erased_element(image_array, self.blur_intensity)
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image)
            results[i] = image/255
        return results


if __name__ == "__main__":

    # Exemple d'utilisation
    path = "degradations/datasets/original/"  
    img = cv2.imread(path+"FRAN_0568_000019_L.jpg")
    erased_img = erased_element(img)

    # resize image for better visualization
    scale_percent = 10 # percent of original size
    width = int(erased_img.shape[1] * scale_percent / 100)
    height = int(erased_img.shape[0] * scale_percent / 100)
    dim = (width, height)
    erased_img = cv2.resize(erased_img, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow("Erased Effect", erased_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
