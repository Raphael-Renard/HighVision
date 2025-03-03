import cv2
import numpy as np
import random
import string

def text_overlay(img):
    """
    Add two lines of random black text on a white rectangle in one corner or at the top of the image
    """
    # pick a corner of the image at random (where the text will be)
    corner = np.random.randint(0, 5)

    # corner that will be overlayed by text
    if corner==4: # text at the top
        ratio_height, ratio_width = np.random.uniform(0.4, 0.6),np.random.uniform(0.2, 0.3)
    else: # in one corner
        ratio_height, ratio_width = np.random.uniform(0.4, 0.7),np.random.uniform(0.7, 0.9)
    half_height, half_width = img.shape[0] // 2, img.shape[1] // 2
    corner_height, corner_width = int(half_height*ratio_height), int(half_width*ratio_width)
    border = img.shape[0]//30

    # text
    font_size_1 = corner_height//85
    thickness_1 = 3*font_size_1
    font_size_2 = font_size_1//2
    thickness_2 = 4*font_size_2

    if corner==4: # top center
        length_txt1 = (img.shape[1]-2*corner_width)//400
        length_txt2 = (img.shape[1]-2*corner_width)//250
    else: # corner
        length_txt1 = corner_width//400
        length_txt2 = corner_width//200

    text1 = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase) for _ in range(length_txt1)).strip()
    text2 = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase) for _ in range(length_txt2)).strip()


    
    if corner == 0: # top left corner
        img[0:corner_height, 0:corner_width] = 255

        # write random text in the white rectangle
        cv2.putText(img,text1,(0,corner_height//3+ border//2),
                    cv2.FONT_HERSHEY_COMPLEX,font_size_1,(0,0,0),thickness_1,cv2.LINE_AA)
        cv2.putText(img,text2,(0,2*corner_height//3),
                    cv2.FONT_HERSHEY_COMPLEX,font_size_2,(0,0,0),thickness_2,cv2.LINE_AA)

        
    elif corner == 1: # top right corner
        img[0:corner_height, img.shape[1] - corner_width:img.shape[1]] = 255

        # write random text in the white rectangle
        cv2.putText(img,text1,(img.shape[1] - corner_width + border, corner_height//3+ border//2),
                    cv2.FONT_HERSHEY_COMPLEX,font_size_1,(0,0,0),thickness_1,cv2.LINE_AA)
        cv2.putText(img,text2,(img.shape[1] - corner_width + border*2, 2*corner_height//3),
                    cv2.FONT_HERSHEY_COMPLEX,font_size_2,(0,0,0),thickness_2,cv2.LINE_AA)

     
    elif corner == 2: # bottom right corner
        img[img.shape[0] - corner_height:img.shape[0], img.shape[1] - corner_width:img.shape[1]] = 255

        # write random text in the white rectangle
        cv2.putText(img,text1,(img.shape[1] - corner_width + border, img.shape[0] - corner_height + corner_height//3 + border//2),
                    cv2.FONT_HERSHEY_COMPLEX,font_size_1,(0,0,0),thickness_1,cv2.LINE_AA)
        cv2.putText(img,text2,(img.shape[1] - corner_width + border*2, img.shape[0] - corner_height + 2*corner_height//3),
                    cv2.FONT_HERSHEY_COMPLEX,font_size_2,(0,0,0),thickness_2,cv2.LINE_AA)


    elif corner == 3: # bottom left corner
        img[img.shape[0] - corner_height:img.shape[0], 0:corner_width] = 255

        # write random text in the white rectangle
        cv2.putText(img,text1,(0,img.shape[0] - corner_height + corner_height//3 + border//2),
                    cv2.FONT_HERSHEY_COMPLEX,font_size_1,(0,0,0),thickness_1,cv2.LINE_AA)
        cv2.putText(img,text2,(0,img.shape[0] - corner_height+ 2*corner_height//3),
                    cv2.FONT_HERSHEY_COMPLEX,font_size_2,(0,0,0),thickness_2,cv2.LINE_AA)
    
    else: # center
        img[0:corner_height, corner_width : img.shape[1] - corner_width] = 255

        # write random text in the white rectangle
        cv2.putText(img,text1,(corner_width + border, corner_height//3+ border//2),
                    cv2.FONT_HERSHEY_COMPLEX,font_size_1,(0,0,0),thickness_1,cv2.LINE_AA)
        cv2.putText(img,text2,(corner_width + border*2, 2*corner_height//3),
                    cv2.FONT_HERSHEY_COMPLEX,font_size_2,(0,0,0),thickness_2,cv2.LINE_AA)

    return img


import torch.nn as nn
class transforms_text_overlay(nn.Module):
    def __init__(self):
        super(transforms_text_overlay, self).__init__()

    def __call__(self, batch):
        for image in batch:
            image = text_overlay(image)
        return batch




if __name__ == "__main__":
    path = "degradations/datasets/original/"
    img = cv2.imread(path+"FRAN_0568_000020_L.jpg")
    img = text_overlay(img)

    # resize image for better visualization
    scale_percent = 10 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow("corner", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
