#TODO: #5 Metadata creation (OvO - One v One + OvR - One v Rest) + Random Forest classification
#In the paper they use GradientBoosting for metadata creation
from sklearn.ensemble import GradientBoostingClassifier as gb
from sklearn.multiclass import OneVsRestClassifier as ovr, OneVsOneClassifier as ovo 
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import seaborn as sns
from typing import Tuple
import pickle as pkl
from sklearn.metrics import confusion_matrix

def meta_data_creation_train(ovo, ovr, kfold : KFold, X : np.array, y : np.array):
    '''
    This function creates the metadata for the training set using the k-fold cross validation.

    Parameters:
    ovo: One vs One classifier
    ovr: One vs Rest classifier
    kfold: KFold object
    X: training set
    y: metalabels of the training set

    Returns:
    meta_data: metadata of the training set
    actual_classes: actual metalabel of the training set
    actual_trainset: actual training set
    '''
    ovo_ = cp.deepcopy(ovo)
    ovr_ = cp.deepcopy(ovr)
    actual_classes = np.empty([0], dtype=int)
    actual_trainset = np.empty([0, X.shape[1]], dtype=float)
    ovo_m = []
    ovr_m = []
    print("Starting k-fold cross validation...\n")
    n = 0
    for train_ndx, test_ndx in kfold.split(X):
        scores_ovo = []
        scores_ovr = []
        print("Fold {}".format(n+1))
        n = n+1
        train_X, train_y, test_X, test_y = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]

        actual_classes = np.append(actual_classes, test_y)
        actual_trainset = np.append(actual_trainset, test_X, axis=0)
        #train
        ovo_.fit(train_X, train_y)
        print("ovo done")
        ovr_.fit(train_X, train_y)
        print("ovr done")
        
        models_ovo = list(ovo_.estimators_)
        models_ovr = list(ovr_.estimators_)

        for mod in models_ovo:
            scores_ovo.append(mod.predict(test_X))
        for mod in models_ovr:
            scores_ovr.append(mod.predict(test_X))
        if n == 1:
            ovo_m = np.array(scores_ovo).T
            ovr_m = np.array(scores_ovr).T
        else:
            ovo_m = np.vstack([ovo_m, np.array(scores_ovo).T])
            ovr_m = np.vstack([ovr_m, np.array(scores_ovr).T])
        print("Metadata for {}° split created".format(n))
    meta_data = np.hstack([ovo_m, ovr_m])
    print("Metadata for the whole dataset created!")
    return meta_data, actual_classes, actual_trainset

#Define a method to train the models on the whole training set without cross_validation

def train_ovo_ovr(ovo, ovr, X : np.array, y : np.array):
    '''
    This function train the ovo and ovr on the whole training set.
    The new models will be used to create the metadata for the test set.

    Parameters:
    ovo: One vs One classifier
    ovr: One vs Rest classifier
    X: training set
    y: metalabels of the training set

    Returns:
    ovo_: One vs One classifier trained on the whole training set
    ovr_: One vs Rest classifier trained on the whole training set
    '''
    ovo_ = cp.deepcopy(ovo)
    ovr_ = cp.deepcopy(ovr)
    print("Starting training set metadata creation...\n")
    ovo_.fit(X, y)
    print("ovo done")
    ovr_.fit(X, y)
    print("ovr done")
        

    return ovo_, ovr_

def meta_data_creation_test(ovo, ovr, X : np.array, y : np.array):
    
    ovo_ = cp.deepcopy(ovo)
    ovr_ = cp.deepcopy(ovr)
    
    ovo_m = []
    ovr_m = []
    print("Starting test set metadata creation...\n")
    #ovo and ovr are already trained
    models_ovo = list(ovo_.estimators_)
    models_ovr = list(ovr_.estimators_)

    for mod in models_ovo:
        ovo_m.append(mod.predict(X))
    for mod in models_ovr:
        ovr_m.append(mod.predict(X))
    ovo_m = np.array(ovo_m).T
    ovr_m = np.array(ovr_m).T
    print("Metadata for the whole dataset created!")
    meta_data = np.hstack([ovo_m, ovr_m])

    return meta_data

def save_ovo_ovr(ovo, ovr, name=""):
    print("Saving models...")
    with open("src/models/metadata/ovo{}.pkl".format(name), "wb") as f:
        pkl.dump(ovo, f)
    with open("src/models/metadata/ovr{}.pkl".format(name), "wb") as f:
        pkl.dump(ovr, f)
    print("Complete!")


def save_metadata(metadata, split = "train", metalabel=None, train_data=None, name=""):
    print("Saving metadata...")
    with open("data/metadata/metadata{}_{}.pkl".format(name, split), "wb") as f:
        pkl.dump(metadata, f)
    if split == "train":
        if metalabel is None or train_data is None:
            print("Error: metalabel or train_data is None")
            return
        with open("data/metadata/metalabel{}_{}.pkl".format(name, split), "wb") as f:
            pkl.dump(metalabel, f)

        with open("data/metadata/superclass{}_trainset.pkl".format(name), "wb") as f:
            pkl.dump(train_data, f)


    print("Complete!")

def load_metadata(split="train" ,name=""):
    metadata = []
    metalabel = []
    train_data = []

    with open("data/metadata/metadata{}_{}.pkl".format(name, split), "rb") as f:
        metadata = pkl.load(f)
    if split == "train":
        if metalabel is None or train_data is None:
            print("Error: metalabel or train_data is None")
            return
        with open("data/metadata/metalabel{}_{}.pkl".format(name, split), "rb") as f:
            metalabel = pkl.load(f)
        
        with open("data/metadata/superclass{}_trainset.pkl".format(name), "rb") as f:
            train_data = pkl.load(f)

    return metadata, metalabel, train_data