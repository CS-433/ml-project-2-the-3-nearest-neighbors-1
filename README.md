# CS 433 - Machine Learning Class Project 1 : Road Segmentation
## Team name : The 3 Nearest Neighbors
### Authors : BOYER Benjamin, GOYBET François, NIELLY Gauthier

This repository contains all the source codes and results files we created for this project :

```
│   .gitignore
│   main.py
│   README.md
│   requirements.txt
│
├───data
│   │
│   ├───test_set_images
│   │
│   └───training
│       ├───groundtruth
│       │
│       └───images
│
├───notebooks
│       segment_aerial_images.ipynb
│
└───src
    │   helper.py
    │   RoadSegmentationDataset.py
    │
    └───models
            CNN.py
            LogReg_Patch.py
            MLP.py
            MLP_Patch.py
            RoadBackgroundClassifierCNN.py
            trainer.py
            Unet.py
            ViT.py
            __init__.py
```



To retrieve our final results, you need first to install the necessary libraries from the *requirements* file :
```
$ pip install -r requirements.txt
```
Given the size of the project, you can also create a specific execution environment and pip install the libraries on it, e.g. with ***conda*** :
```
conda create -n roadsegmentation pip
```

Then, you need to download the project dataset and store it in the same repertory, under :
```
data\
```

Finally, you can run the main.py script :
```
python main.py --train <list of models> (optional)
```
The ***train*** argument allows you to choose which model(s) you want to train. For example, if you to try out our patch-splitting approach with a Logistic Regression model and our holistic solution with a U Network, you can run with : *--train patch_logreg holistic_unet*

> **_NOTE:_**  We have achieved our final performances with a P100 GPU from **Kaggle**. However we have adapted hyperparameters in the *main.py* so that all the models can be trained on a CPU. If is then very likely that they will be less efficient than the ones we discuss in our report.

Here are the steps of the execution :
1. Load and preprocess data
2. For each model specified after the --train argument : instantiate, fit, save and finally validate the model
3. For the other models : load pre-trained and compute scores on the validation set
4. Predict the segmentation masks of the test set and create a submission for the best model
