from src.helper import *
import os
import numpy as np
from sklearn.model_selection import train_test_split

TRAINING_PATH_RS = "training/"
TRAINING_IMAGE_DIR_RS = TRAINING_PATH_RS + "images/"
TRAINING_GT_DIR_RS = TRAINING_PATH_RS + "groundtruth/"

TRAINING_PATH_GM = "train/"
TRAINING_IMAGE_DIR_GM = TRAINING_PATH_GM + "images/"
TRAINING_GT_DIR_GM = TRAINING_PATH_GM + "label/"

NB_IMAGES_TRAINING_RS = 100
NB_IMAGES_TRAINING_GM = 50

FOREGROUND_THRESHOLD = 0.25

class RoadSegmentationDataset:
    """
    This class contains the images from the original roagsegmentation dataset, and some from googlemap.
    """
    def __init__(
            self,
            data_path_rs,
            data_path_gm,
            img_size=400,
            nb_image_training_rs=100,
            nb_image_training_gm=50,
            ):

        assert 400 % img_size == 0, f"Can't divide 400 by {img_size}"

        #road-segmentation
        files = os.listdir(data_path_rs + TRAINING_IMAGE_DIR_RS)
        nb_images = min(nb_image_training_rs, len(files))
        
        self.imgs_rs = [load_image(data_path_rs + TRAINING_IMAGE_DIR_RS + file) for file in files[:nb_images]]
        self.gt_imgs_rs = [load_image(data_path_rs + TRAINING_GT_DIR_RS + file) for file in files[:nb_images]]

        # google-map
        files = os.listdir(data_path_gm + TRAINING_IMAGE_DIR_GM)
        nb_images = min(nb_image_training_gm, len(files))

        self.imgs_gm = [load_image(data_path_gm + TRAINING_IMAGE_DIR_GM + file) for file in files[:nb_images]]
        self.gt_imgs_gm = [load_image(data_path_gm + TRAINING_GT_DIR_GM + file) for file in files[:nb_images]]

        self.gt_imgs_gm = np.max(np.array(self.gt_imgs_gm), axis=-1, keepdims=True)
        
        self.imgs_gm = [resize_image(np.array(img), new_size=img_size) for img in self.imgs_gm]
        self.gt_imgs_gm = [resize_image(np.array(gt_img), new_size=img_size) for gt_img in self.gt_imgs_gm]


    def get_XY(self, val_size=0.2, include_gm=False, patch_size=None):

        X = self.imgs_rs
        y = self.gt_imgs_rs
        
        if include_gm:
            X += self.imgs_gm
            y += self.gt_imgs_gm

        if patch_size:
            X = [img_crop(img, patch_size, patch_size) for img in X]
            X = [X[i][j] for i in range(len(X)) for j in range(len(X[i]))]
            
            y = [img_crop(img, patch_size, patch_size) for img in y]
            y = [y[i][j] for i in range(len(y)) for j in range(len(y[i]))]
            y = [value_to_class(img, FOREGROUND_THRESHOLD) for img in y]

        X = np.array(X).transpose(0,3,1,2) # convert in array of shape (nb_images, 3, img_size, img_size)
        y = (np.expand_dims(np.array(y), axis=1) >= 0.5).astype(float) # convert in binary array of shape (nb_images, 1, img_size, img_size)
        return train_test_split(X, y, test_size=val_size, random_state=42)
    
