from src.preprocessing.preprocessing import *
import os
import numpy as np
from sklearn.model_selection import train_test_split

TRAINING_PATH = "training/"
TRAINING_IMAGE_DIR = TRAINING_PATH + "images/"
TRAINING_GT_DIR = TRAINING_PATH + "groundtruth/"

class RoadSegmentationDataset():

    def __init__(
            self,
            data_path,
            img_size=400,
            nb_image_training=100,
            random_shift=False,
            scramble_image=False,
            noise=0.0,
            ):
        
        assert 400 % img_size == 0, f"Can't divide 400 by {img_size}"
        self.img_size = img_size
        
        files = os.listdir(data_path + TRAINING_IMAGE_DIR)
        nb_images = min(nb_image_training, len(files))
        
        self.imgs = [load_image(data_path + TRAINING_IMAGE_DIR + file) for file in files[:nb_images]]
        self.gt_imgs = [load_image(data_path + TRAINING_GT_DIR + file) for file in files[:nb_images]]
        
        self.imgs = np.array(self.imgs).transpose(0,3,1,2)
        self.gt_imgs = np.expand_dims(np.array(self.gt_imgs), axis=1) >= .5

    def get_XY(self, test_size=0.2):
        return train_test_split(self.imgs, self.gt_imgs, test_size=test_size,random_state=42)