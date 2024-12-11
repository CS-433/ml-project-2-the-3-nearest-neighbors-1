import matplotlib.image as mpimg
import numpy as np

def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j : j + w, i : i + h]
            else:
                im_patch = im[j : j + w, i : i + h, :]
            list_patches.append(im_patch)
    return list_patches

def concat_images(img_rgb, img_gray1, img_gray2):
    
    if img_gray1.shape[1] == 1:
        img_gray1 = np.repeat(img_gray1, 3, axis=1)  
    if img_gray2.shape[1] == 1:
        img_gray2 = np.repeat(img_gray2, 3, axis=1) 

    concatenated_img = np.concatenate((img_rgb, img_gray1, img_gray2), axis=3)
    concatenated_img = np.transpose(concatenated_img, (0, 2, 3, 1))
    return concatenated_img