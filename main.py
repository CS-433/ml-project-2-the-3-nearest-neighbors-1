import argparse
import os, sys
from src.preprocessing.preprocessing import *
import matplotlib.pyplot as plt
from src.models.MLP import MLP
from src.models.Unet import UNet
from src.models.CNN import RoadSegmentationCNN
from src.models.trainer import Trainer
from src.RoadSegmentationDataset import RoadSegmentationDataset
import numpy as np

DATA_PATH = "data/" 
NB_IMAGES_TRAINING = 20

def main(args):
    
    dataset = RoadSegmentationDataset(DATA_PATH, nb_image_training=NB_IMAGES_TRAINING)
    X_train, X_test, y_train, y_test = dataset.get_XY()
    model = MLP()
    trainer = Trainer(model=model, lr=0.01, epochs=10, batch_size=100)
    trainer.fit(X_train,y_train)

    score_train = trainer.score(y_true=y_train, y_pred=trainer.predict(X_train))
    score_test = trainer.score(y_true=y_test, y_pred=trainer.predict(X_test))
    print(score_train)
    print(score_test)

    imgs_show = [0,1,2]
    conc_img = concat_images(X_train[imgs_show], y_train[imgs_show], trainer.predict(X_train[imgs_show]) >= .5)
    plt.imshow(conc_img[0])
    plt.show()



if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(
        description="Run model with optional cross validation"
    )

    args = parser.parse_args()

    main(args)