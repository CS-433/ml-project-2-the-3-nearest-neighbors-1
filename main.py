import argparse
import os, sys
from src.helper import *
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

NB_IMAGES_TRAINING_RS = 50
NB_IMAGES_TRAINING_GM = 2
BEST_THRESHOLD = 0.75

GENERATED_MODELS_PATH = "generated/models/"

def main(args):

    # Load Dataset
    dataset = RoadSegmentationDataset(
        DATA_PATH_RS, 
        DATA_PATH_GM, 
        nb_image_training_rs=NB_IMAGES_TRAINING_RS,
        nb_image_training_gm=NB_IMAGES_TRAINING_GM,
    )

    # ##############################
    # #######     Patch      #######  
    # ##############################  
    # X_train, X_val, y_train, y_val = dataset.get_XY(val_size=0.10, include_gm=False, patch_size=16)
    
    # ####################
    # ### Patch LogReg ###
    # ####################
    # print("patch_logreg :")
    # if "patch_logreg" in args.train:
    #     print("[training and saving model]")
    #     model_logReg = LogReg_patch()
    #     model_logReg.fit(X_train, y_train)
    #     joblib.dump(model_logReg,GENERATED_MODELS_PATH+"patch_logreg")

    # else :
    #     print("[loading the model]")
    #     model_logReg = joblib.load(GENERATED_MODELS_PATH + "patch_logreg")
    
    # score_logReg_patch = score(y_val, model_logReg.predict(X_val))
    # with open(GENERATED_MODELS_PATH + "Patch_LogReg_result.txt", "w") as f:
    #     f.write(str(score_logReg_patch))
    # print(score_logReg_patch)

    # ####################
    # ###  Patch MLP   ###
    # ####################
    # print("patch_mlp :")
    # if "patch_mlp" in args.train:
    #     print("[training and saving model]")
    #     model_MLP_patch = MLP_patch()
    #     model_MLP_patch.fit(X_train, y_train)
    #     joblib.dump(model_MLP_patch,GENERATED_MODELS_PATH+"patch_mlp")

    # else : 
    #     print("[loading the model]")
    #     model_MLP_patch = joblib.load(GENERATED_MODELS_PATH + "patch_mlp")

    # score_MLP_patch = score(y_val, model_MLP_patch.predict(X_val))
    # print(score_MLP_patch)
    # with open(GENERATED_MODELS_PATH + "Patch_MLP_result.txt", "w") as f:
    #     f.write(str(score_MLP_patch))

    # ####################
    # ###  Patch CNN   ###
    # ####################
    # print("patch_cnn :")
    # if "patch_cnn" in args.train:
    #     print("[training and saving model]")
    #     model_CNN_patch = RoadBackgroundClassifierCNN()
    #     trainer_CNN_patch = Trainer(
    #         model=model_CNN_patch, 
    #         lr=5e-4, 
    #         weight_decay=1e-4, 
    #         epochs=50, 
    #         batch_size=128, 
    #     )
    #     history_CNN_patch = trainer_CNN_patch.fit(X_train,y_train, X_val, y_val)
    #     joblib.dump(trainer_CNN_patch,GENERATED_MODELS_PATH + "patch_cnn_epoch_50_batch_128")
    
    # else :
    #     print("[loading the model]")
    #     trainer_CNN_patch = joblib.load(GENERATED_MODELS_PATH + "patch_cnn_epoch_50_batch_128")

    # score_CNN_patch = trainer_CNN_patch.score(y_val, trainer_CNN_patch.predict(X_val)>=.5)
    # print(score_CNN_patch)
    # with open(GENERATED_MODELS_PATH + "Patch_CNN_result.txt", "w") as f:
    #     f.write(str(score_CNN_patch))


    # ##############################
    # #######   Holistic     #######  
    # ##############################  
    X_train, X_val, y_train, y_val = dataset.get_XY(val_size=0.10, include_gm=False)
    # ####################
    # ### Holistic MLP ###
    # ####################
    # print("holistic_mlp :")
    # if "holistic_mlp" in args.train:
    #     print("[training and saving model]")
    #     model = MLP()
    #     num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     print("number of trainable parameters :", num_params)
    #     trainer_MLP_holistic = Trainer(
    #         model=model, 
    #         lr=5e-4, 
    #         weight_decay=1e-4, 
    #         epochs=50, 
    #         batch_size=5, 
    #     )
    #     history_MLP_holistic = trainer_MLP_holistic.fit(X_train,y_train, X_val, y_val)
    #     joblib.dump(trainer_MLP_holistic,GENERATED_MODELS_PATH+"holistic_mlp_epoch_50_batch_5")
    
    # else :
    #     print("[loading the model]")
    #     trainer_MLP_holistic = joblib.load(GENERATED_MODELS_PATH + "holistic_mlp_epoch_50_batch_5")

    # score_MLP_holistic = trainer_MLP_holistic.score(y_val, trainer_MLP_holistic.predict(X_val)>=.5)
    # print(score_MLP_holistic)
    # with open(GENERATED_MODELS_PATH + "Holistic_MLP_result.txt", "w") as f:
    #     f.write(str(score_MLP_holistic))

    # ####################
    # ### Holistic CNN ###
    # ####################
    # print("holistic_cnn :")
    # if "holistic_cnn" in args.train:
    #     print("[training and saving model]")
    #     model = EncoderDecoderCNN()
    #     num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     print("number of trainable parameters :", num_params)
    #     trainer_CNN_holistic = Trainer(
    #         model=model, 
    #         lr=5e-4, 
    #         weight_decay=1e-4, 
    #         epochs=50, 
    #         batch_size=5, 
    #     )
    #     history_CNN_holistic = trainer_CNN_holistic.fit(X_train,y_train, X_val, y_val)
    #     joblib.dump(trainer_CNN_holistic,GENERATED_MODELS_PATH+"holistic_cnn_epoch_50_batch_5")

    # else :
    #     print("[loading the model]")
    #     trainer_CNN_holistic = joblib.load(GENERATED_MODELS_PATH + "holistic_cnn_epoch_50_batch_5")

    # score_CNN_holistic = trainer_CNN_holistic.score(y_val, trainer_CNN_holistic.predict(X_val)>=.5)
    # print(score_CNN_holistic)
    # with open(GENERATED_MODELS_PATH + "Holistic_CNN_result.txt", "w") as f:
    #     f.write(str(score_CNN_holistic))

    ####################
    ### Holistic Unet###
    ####################
    print("holistic_unet :")
    if "holistic_unet" in args.train:
        print("[training and saving model]")
        model = UNet()
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("number of trainable parameters :", num_params)
        trainer_UNET_holistic = Trainer(
            model=model, 
            lr=5e-4, 
            weight_decay=1e-4, 
            epochs=50, 
            batch_size=5, 
        )
        history_UNET_holistic = trainer_UNET_holistic.fit(X_train,y_train, X_val, y_val)
        joblib.dump(trainer_UNET_holistic,GENERATED_MODELS_PATH+"holistic_unet_epoch_50_batch_5_rs")

    else :
        print("[loading the model]")
        trainer_UNET_holistic = joblib.load(GENERATED_MODELS_PATH + "holistic_unet_epoch_50_batch_5_rs")
    
    score_UNET_holistic = trainer_UNET_holistic.score(y_val, trainer_UNET_holistic.predict(X_val)>=.5)
    print(score_UNET_holistic)
    with open(GENERATED_MODELS_PATH + "Holistic_UNet_result.txt", "w") as f:
        f.write(str(score_UNET_holistic))

    X_test = dataset.get_test()
    y_pred = (trainer_UNET_holistic.predict(X_test) >= BEST_THRESHOLD).astype(float)
    masks_to_submission(GENERATED_MODELS_PATH + "unet_submission.csv", y_pred)


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(
        description="Run model with optional cross-validation"
    )
    parser.add_argument(
        "--train", 
        nargs="+", 
        default=[], 
        help="List of models to train (e.g., --train patch_logreg patch_mlp patch_cnn holistic_mlp holistic_cnn holistic_unet"
    )
    
    args = parser.parse_args()
    main(args)