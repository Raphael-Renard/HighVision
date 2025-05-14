import cv2
import numpy as np
import random
import string
import torch
import torch.nn as nn
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from degradations.methods.utils import remove_black_borders, restore_black_borders


def text_around(img):
    nb_text = np.random.randint(1, 5)

    # add white borders
    h, w = img.shape[0], img.shape[1]
    border_t, border_b = np.random.randint(h//10, h//4), np.random.randint(h//10, h//4)
    border_l, border_r = np.random.randint(w//10, w//6), np.random.randint(w//10, w//6)

    border_h = max(border_t, border_b)
    border_w = max(border_l, border_r)

    img = cv2.copyMakeBorder(img, border_t, border_b, border_l, border_r, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    h_new, w_new = img.shape[0], img.shape[1]

    for _ in range(nb_text):
        # add random text in the borders
        length_txt = random.randint(border_w//100, border_w//50)
        if w < 300:
            font_size = 1
            thickness = 1
        else:
            font_size = random.randint(border_h//100, border_h//50)
            thickness = np.random.randint(1, 30)

        text = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase) for _ in range(length_txt)).strip()

        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, font_size, thickness)
        text_width, text_height = text_size

        while True:
            x = random.randint(0, w_new - text_width)
            y = random.randint(text_height, h_new)
            if (x + text_width < border_l) or x > border_l + w \
                or (y + text_height < border_t ) or y > border_t + h:
                break

        cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_COMPLEX, font_size, (0,0,0), thickness, cv2.LINE_AA)

    img = cv2.resize(img, (w, h))    
    
    return img






class transforms_text_around(nn.Module):
    def __init__(self):
        super(transforms_text_around, self).__init__()

    def __call__(self, batch):
        one_image = False
        if len(batch.shape) == 3: # si on ne passe qu'une image au lieu d'un batch
            batch = batch.unsqueeze(0)
            one_image = True
            
        results = torch.empty_like(batch)
        for i, image in enumerate(batch):
            image_array = np.transpose(np.array(image), (1, 2, 0)).copy() * 255
            image_array = image_array.astype(np.uint8)
            shape = image_array.shape
            image, black_borders = remove_black_borders(image_array)

            image = text_around(image) 

            image = restore_black_borders(shape, image, black_borders)
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image) / 255
            results[i] = image
        
        if one_image:
            results = results.squeeze(0)
        return results

