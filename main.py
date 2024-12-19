import argparse
import os, sys
from src.preprocessing.preprocessing import *
import matplotlib.pyplot as plt
from src.models import *
from src.RoadSegmentationDataset import RoadSegmentationDataset
import numpy as np
import joblib

DATA_PATH_RS = "data/rs/"
TRAINING_PATH_RS = "training/"
TRAINING_IMAGE_DIR_RS = TRAINING_PATH_RS + "images/"
TRAINING_GT_DIR_RS = TRAINING_PATH_RS + "groundtruth/"

DATA_PATH_GM = "data/gm/"
TRAINING_PATH_GM = "train/"
TRAINING_IMAGE_DIR_GM = TRAINING_PATH_GM + "images/"
TRAINING_GT_DIR_GM = TRAINING_PATH_GM + "label/"

NB_IMAGES_TRAINING_RS = 2
NB_IMAGES_TRAINING_GM = 2

GENERATED_MODELS_PATH = "generated/models/"

def main(args):
        
    # Load Dataset
    dataset = RoadSegmentationDataset(
        DATA_PATH_RS, 
        DATA_PATH_GM, 
        nb_image_training_rs=NB_IMAGES_TRAINING_RS,
        nb_image_training_gm=NB_IMAGES_TRAINING_GM,
    )


    # Patch
    X_train, X_val, y_train, y_val = dataset.get_XY(val_size=0.10, include_gm=False, patch_size=16)

    # Patch : LogReg
    print("Patch : LogReg")
    model_logReg = LogReg_patch()
    model_logReg.fit(X_train, y_train)
    score_logReg_patch = score(y_val, model_logReg.predict(X_val))
    joblib.dump(model_logReg,GENERATED_MODELS_PATH+"patch_logreg")
    print(score_logReg_patch)
    with open(GENERATED_MODELS_PATH + "Patch_LogReg_result.txt", "w") as f:
        f.write(str(score_logReg_patch))


    # Patch : MLP
    print("Patch : MLP")
    model_MLP_patch = MLP_patch()
    model_MLP_patch.fit(X_train, y_train)
    score_MLP_patch = score(y_val, model_MLP_patch.predict(X_val))
    joblib.dump(model_MLP_patch,GENERATED_MODELS_PATH+"patch_mlp")
    print(score_MLP_patch)
    with open(GENERATED_MODELS_PATH + "Patch_MLP_result.txt", "w") as f:
        f.write(str(score_MLP_patch))

    # Patch : CNN
    print("Patch : CNN")
    model_CNN_patch = RoadBackgroundClassifierCNN()
    trainer_CNN_patch = Trainer(
        model=model_CNN_patch, 
        lr=5e-4, 
        weight_decay=1e-4, 
        epochs=20, 
        batch_size=128, 
    )
    history_CNN_patch = trainer_CNN_patch.fit(X_train,y_train, X_val, y_val)
    score_CNN_patch = trainer_CNN_patch.score(y_val, trainer_CNN_patch.predict(X_val)>=.5)
    joblib.dump(trainer_CNN_patch,GENERATED_MODELS_PATH + "patch_cnn_epoch_50_batch_128")
    print(score_CNN_patch)
    with open(GENERATED_MODELS_PATH + "Patch_MLP_result.txt", "w") as f:
        f.write(str(score_CNN_patch))


    # Holistic
    X_train, X_val, y_train, y_val = dataset.get_XY(val_size=0.10, include_gm=False)

    # Holistic : MLP
    print("Holistic : MLP")
    model = MLP()
    trainer_MLP_holistic = Trainer(
        model=model, 
        lr=5e-4, 
        weight_decay=1e-4, 
        epochs=50, 
        batch_size=5, 
    )
    history_MLP_holistic = trainer_MLP_holistic.fit(X_train,y_train, X_val, y_val)
    score_MLP_holistic = trainer_MLP_holistic.score(y_val, trainer_MLP_holistic.predict(X_val)>=.5)
    joblib.dump(trainer_MLP_holistic,GENERATED_MODELS_PATH+"holistic_mlp_epoch_50_batch_5")
    print(score_MLP_holistic)
    with open(GENERATED_MODELS_PATH + "Holistic_MLP_result.txt", "w") as f:
        f.write(str(score_MLP_holistic))

    # Holistic : CNN
    print("Holistic : CNN")
    model = EncoderDecoderCNN()
    trainer_CNN_holistic = Trainer(
        model=model, 
        lr=5e-4, 
        weight_decay=1e-4, 
        epochs=50, 
        batch_size=5, 
    )
    history_CNN_holistic = trainer_CNN_holistic.fit(X_train,y_train, X_val, y_val)
    score_CNN_holistic = trainer_CNN_holistic.score(y_val, trainer_CNN_holistic.predict(X_val)>=.5)
    joblib.dump(trainer_CNN_holistic,GENERATED_MODELS_PATH+"holistic_cnn_epoch_50_batch_5")
    print(score_CNN_holistic)
    with open(GENERATED_MODELS_PATH + "Holistic_CNN_result.txt", "w") as f:
        f.write(str(score_CNN_holistic))

    # Holistic : Unet
    print("Holistic : Unet")
    model = UNet()
    trainer_UNET_holistic = Trainer(
        model=model, 
        lr=5e-4, 
        weight_decay=1e-4, 
        epochs=50, 
        batch_size=5, 
    )
    history_UNET_holistic = trainer_UNET_holistic.fit(X_train,y_train, X_val, y_val)
    score_UNET_holistic = trainer_UNET_holistic.score(y_val, trainer_UNET_holistic.predict(X_val)>=.5)
    joblib.dump(trainer_UNET_holistic,GENERATED_MODELS_PATH+"holistic_unet_epoch_50_batch_5_rs")
    print(score_UNET_holistic)
    with open(GENERATED_MODELS_PATH + "Holistic_UNet_result.txt", "w") as f:
        f.write(str(score_UNET_holistic))


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(
        description="Run model with optional cross validation"
    )

    args = parser.parse_args()

    main(args)