import argparse
import os, sys
from src.data.preprocessing import *
import matplotlib.pyplot as plt
from src.models.MLP import MLP 
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
    
    X_train = np.array(imgs)
    y_train = np.array(gt_imgs)

    X_train = X_train.reshape((X_train.shape[0],-1))
    y_train = y_train.reshape((y_train.shape[0],-1))

    model = MLP()
    trainer = Trainer(model=model, lr=0.01, epochs=10, batch_size=12)
    trainer.fit(X_train,y_train)
    
    example = trainer.predict(np.array(X_train[:1]))
    print(example.shape)
    plt.imshow(example.reshape((400,400)))
    plt.show()


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(
        description="Run model with optional cross validation"
    )

    args = parser.parse_args()

    main(args)