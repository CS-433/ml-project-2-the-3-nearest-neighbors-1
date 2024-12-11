import argparse
import os, sys
from src.data.preprocessing import *
import matplotlib.pyplot as plt
from src.models.MLP import MLP
from src.models.Unet import UNet
from src.models.CNN import RoadSegmentationCNN
from src.models.trainer import Trainer
import numpy as np

DATA_PATH = "data/" 
TRAINING_PATH = DATA_PATH + "training/"
TRAINING_IMAGE_DIR = TRAINING_PATH + "images/"
TRAINING_GT_DIR = TRAINING_PATH + "groundtruth/"
NB_IMAGES_TRAINING = 20

def main(args):
    
    files = os.listdir(TRAINING_IMAGE_DIR)
    nb_images = min(NB_IMAGES_TRAINING, len(files))
    
    imgs = [load_image(TRAINING_IMAGE_DIR + file) for file in files[:nb_images]]
    gt_imgs = [load_image(TRAINING_GT_DIR + file) for file in files[:nb_images]]
    
    patch_size = 1
    size_image = 400 // patch_size

    # imgs = [img_crop(imgs[i], patch_size, patch_size) for i in range(nb_images)]
    # gt_imgs = [img_crop(gt_imgs[i], patch_size, patch_size) for i in range(nb_images)]
    
    X_train = np.array(imgs)
    print(X_train.shape)
    y_train = np.array(gt_imgs)
    print(y_train.shape)

    # X_train = X_train.reshape((X_train.shape[0],-1))
    # y_train = y_train.reshape((y_train.shape[0],-1))

    model = RoadSegmentationCNN()
    trainer = Trainer(model=model, lr=0.01, epochs=2, batch_size=12)
    trainer.fit(X_train,y_train)
    
    example = trainer.predict(np.array(X_train[:1])) 
    print(example.shape)
    print(example)
    print(example >= .5)
    print(np.array(y_train[:1]).reshape((size_image,size_image)))

    f, axarr = plt.subplots(3,1) 

    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    axarr[0].imshow(np.array(X_train[:1]).reshape((size_image,size_image,3)))
    axarr[1].imshow(example.reshape((size_image,size_image)))
    axarr[2].imshow(np.array(y_train[:1]).reshape((size_image,size_image)))
    plt.show()


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(
        description="Run model with optional cross validation"
    )

    args = parser.parse_args()

    main(args)