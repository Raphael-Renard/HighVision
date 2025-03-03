import cv2
import numpy as np

def picture_overlay(img, img2=None):
    "Add part of another picture in the corner of the image"

    if img2 == None:
        img2 = img.copy()

    img_copy = img.copy()

    # corner of the second image to overlay
    ratio_height, ratio_width = np.random.uniform(0.2, 0.5),np.random.uniform(0.2, 0.5) # part of the corner of the second picture to overlay
    half_height, half_width = img2.shape[0] // 2, img2.shape[1] // 2
    corner_height, corner_width = int(half_height*ratio_height), int(half_width*ratio_width)

    # pick a corner of the first image at random (where the second image will be overlayed)
    corner = np.random.randint(0, 4)

    thickness = max(img.shape[0], img.shape[1]) // 150 # separate the two images with a white line of this thickness


    cut_corner = np.random.randint(0, 2) # is the corner of the second image cut or not
 
    if cut_corner == 1:
        cut_height, cut_width = int(corner_height*np.random.uniform(0.3, 0.5)), int(corner_width*np.random.uniform(0.3, 0.5))


    # overlay a corner of the second image on the first image
    

    if corner == 0: # top left corner
        img[0:corner_height, 0:corner_width] = img2[-corner_height:,-corner_width:]

        if cut_corner == 1:
            img[cut_height:corner_height, cut_width:corner_width] = img_copy[cut_height:corner_height, cut_width:corner_width]
            
            # add a white line to separate the two images
            cv2.line(img, (cut_width, cut_height), (corner_width, cut_height), (255, 255, 255), thickness)
            cv2.line(img, (cut_width, cut_height), (cut_width, corner_height), (255, 255, 255), thickness)
            cv2.line(img, (cut_width, corner_height), (0, corner_height), (255, 255, 255), thickness)
            cv2.line(img, (corner_width, cut_height), (corner_width, 0), (255, 255, 255), thickness)
        
        else:
            # add a white line to separate the two images
            cv2.line(img, (0, corner_height), (corner_width, corner_height), (255, 255, 255), thickness)
            cv2.line(img, (corner_width, 0), (corner_width, corner_height), (255, 255, 255), thickness)


    elif corner == 1: # top right corner
        img[0:corner_height, img.shape[1] - corner_width:img.shape[1]] = img2[-corner_height:,:corner_width]

        if cut_corner == 1:
            img[cut_height:corner_height, img.shape[1] - corner_width:img.shape[1] - cut_width] = img_copy[cut_height:corner_height, img.shape[1] - corner_width:img.shape[1] - cut_width]

            # add a white line to separate the two images
            cv2.line(img, (img.shape[1] - cut_width, cut_height), (img.shape[1] - corner_width, cut_height), (255, 255, 255), thickness)
            cv2.line(img, (img.shape[1] - cut_width, cut_height), (img.shape[1] - cut_width, corner_height), (255, 255, 255), thickness)
            cv2.line(img, (img.shape[1] - cut_width, corner_height), (img.shape[1], corner_height), (255, 255, 255), thickness)
            cv2.line(img, (img.shape[1] - corner_width, cut_height), (img.shape[1] - corner_width, 0), (255, 255, 255), thickness)

        else:
            # add a white line to separate the two images
            cv2.line(img, (img.shape[1], corner_height), (img.shape[1] - corner_width, corner_height), (255, 255, 255), thickness)
            cv2.line(img, (img.shape[1] - corner_width, 0), (img.shape[1] - corner_width, corner_height), (255, 255, 255), thickness)


    elif corner == 2: # bottom right corner
        img[img.shape[0] - corner_height:img.shape[0], img.shape[1] - corner_width:img.shape[1]] = img2[:corner_height,:corner_width]

        if cut_corner == 1:
            img[img.shape[0] - corner_height:img.shape[0] - cut_height, img.shape[1] - corner_width:img.shape[1] - cut_width] = img_copy[img.shape[0] - corner_height:img.shape[0] - cut_height, img.shape[1] - corner_width:img.shape[1] - cut_width]

            # add a white line to separate the two images
            cv2.line(img, (img.shape[1] - cut_width, img.shape[0] - cut_height), (img.shape[1] - corner_width, img.shape[0] - cut_height), (255, 255, 255), thickness)
            cv2.line(img, (img.shape[1] - cut_width, img.shape[0] - cut_height), (img.shape[1] - cut_width, img.shape[0] - corner_height), (255, 255, 255), thickness)
            cv2.line(img, (img.shape[1] - cut_width, img.shape[0] - corner_height), (img.shape[1], img.shape[0] - corner_height), (255, 255, 255), thickness)
            cv2.line(img, (img.shape[1] - corner_width, img.shape[0] - cut_height), (img.shape[1] - corner_width, img.shape[0]), (255, 255, 255), thickness)

        else:
            # add a white line to separate the two images
            cv2.line(img, (img.shape[1], img.shape[0] - corner_height), (img.shape[1] - corner_width, img.shape[0] - corner_height), (255, 255, 255), thickness)
            cv2.line(img, (img.shape[1] - corner_width, img.shape[0]), (img.shape[1] - corner_width, img.shape[0] - corner_height), (255, 255, 255), thickness)


    else: # bottom left corner
        img[img.shape[0] - corner_height:img.shape[0], 0:corner_width] = img2[:corner_height,-corner_width:]

        if cut_corner == 1:
            img[img.shape[0] - corner_height:img.shape[0] - cut_height, cut_width:corner_width] = img_copy[img.shape[0] - corner_height:img.shape[0] - cut_height, cut_width:corner_width]

            # add a white line to separate the two images
            cv2.line(img, (cut_width, img.shape[0] - cut_height), (corner_width, img.shape[0] - cut_height), (255, 255, 255), thickness)
            cv2.line(img, (cut_width, img.shape[0] - cut_height), (cut_width, img.shape[0] - corner_height), (255, 255, 255), thickness)
            cv2.line(img, (cut_width, img.shape[0] - corner_height), (0, img.shape[0] - corner_height), (255, 255, 255), thickness)
            cv2.line(img, (corner_width, img.shape[0] - cut_height), (corner_width, img.shape[0]), (255, 255, 255), thickness)

        else:
            # add a white line to separate the two images
            cv2.line(img, (0, img.shape[0] - corner_height), (corner_width, img.shape[0] - corner_height), (255, 255, 255), thickness)
            cv2.line(img, (corner_width, img.shape[0]), (corner_width, img.shape[0] - corner_height), (255, 255, 255), thickness)

    return img



import torch.nn as nn
class transforms_picture_overlay(nn.Module):
    def __init__(self):
        super(transforms_picture_overlay, self).__init__()

    def __call__(self, batch):
        for image in batch:
            image = picture_overlay(image)
        return batch




if __name__ == "__main__":
    path = "degradations/datasets/original/"
    img = cv2.imread(path+"FRAN_0568_000183_L.jpg")
    img2 = cv2.imread(path+"/FRAN_0568_000220_L.jpg")
    img = picture_overlay(img, img2)

    # resize image for better visualization
    scale_percent = 10 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow("corner", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
