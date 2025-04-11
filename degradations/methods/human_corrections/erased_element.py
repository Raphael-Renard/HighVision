import cv2
from ultralytics import YOLO
import numpy as np
import torch.nn as nn
import torch
from skimage.restoration import inpaint

def erased_element(img, blur_intensity=101, verbose=False, inpaint_method="telea"):
    """Detects and removes the largest object outside the center region using YOLO."""
    model = YOLO("yolov8n.pt", verbose=False)
    
    # Define center region
    height, width = img.shape[:2]
    center_region = (width * 0.25, height * 0.25, width * 0.75, height * 0.75)

    results = model(img,verbose=False)
    if inpaint_method == "telea" or inpaint_method == "lama":
        mask = np.zeros((height, width), dtype=np.uint8)
    elif inpaint_method=="biharmonic":
        mask = np.zeros((height, width), dtype=bool)
    
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
    if inpaint_method == "telea" or inpaint_method == "lama":
        mask[y1:y2, x1:x2] = 255
    elif inpaint_method == "biharmonic":
        mask[y1:y2, x1:x2] = 1

    # Inpaint the object to erase it
    if inpaint_method == "telea":
        inpainted_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        # add a blur effect to the inpainted region
        inpainted_img[y1:y2, x1:x2] = cv2.GaussianBlur(inpainted_img[y1:y2, x1:x2], (blur_intensity, blur_intensity), 0)

    
    elif inpaint_method == "biharmonic":
        if len(img.shape)==3:
            channel_axis = 2
        else:
            channel_axis = None

        inpainted_img = inpaint.inpaint_biharmonic(img, mask, channel_axis=channel_axis)*255
    
    elif inpaint_method == "lama":
        from lama_cleaner.model_manager import ModelManager
        from lama_cleaner.schema import Config
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = ModelManager(name="lama",device=device,verbose=False) 

        config = Config(
            ldm_steps=25,
            hd_strategy="Resize",
            hd_strategy_crop_margin=32,
            hd_strategy_crop_trigger_size=512,
            hd_strategy_resize_limit=1024,
            )
        inpainted_img = model(img, mask, config).astype(np.uint8)
        inpainted_img = cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2RGB)

    
    return inpainted_img.astype(np.uint8)





class transforms_erased_element(nn.Module):
    def __init__(self, blur_intensity=101):
        super(transforms_erased_element, self).__init__()
        self.blur_intensity = blur_intensity

    def __call__(self, batch):
        one_image = False
        if len(batch.shape) == 3: # si on ne passe qu'une image au lieu d'un batch
            batch = batch.unsqueeze(0)
            one_image = True

        results = torch.empty_like(batch)
        for i, image in enumerate(batch):
            image_array = np.transpose(np.array(image), (1, 2, 0)).copy() * 255
            image_array = image_array.astype(np.uint8)
            image = erased_element(image_array, self.blur_intensity)
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image)
            results[i] = image/255
        
        if one_image:
            results = results.squeeze(0)
        return results


if __name__ =="__main__":
    image_path = "C:/Users/rapha/Documents/Cours/Master/Stage/HighVision/degradations/results/2K2476_16_01.jpg"
    #image_path = "C:/Users/rapha/Documents/Cours/Master/Stage/Data/Sena/FRAN_0568_11AR_699/FRAN_0568_000014_L.jpg"
    img = cv2.imread(image_path)
    img = erased_element(img,inpaint_method="lama")
    cv2.imwrite("erased_element_big.jpg",img)