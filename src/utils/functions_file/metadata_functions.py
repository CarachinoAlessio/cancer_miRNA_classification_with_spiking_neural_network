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

def meta_data_creation_train(ovo, ovr, kfold : KFold, X : np.array, y : np.array) -> Tuple[np.array, np.array, np.array]:

        ovo_ = cp.deepcopy(ovo)
        ovr_ = cp.deepcopy(ovr)
        actual_classes = np.empty([0], dtype=int)
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
        return meta_data, actual_classes, ovo_, ovr_

def save_metadata(metadata, metalabel, name=""):
    print("Saving metadata...")
    with open("data/metadata/metadata{}_train.pkl".format(name), "wb") as f:
        pkl.dump(metadata, f)

    with open("data/metadata/metalabel{}_train.pkl".format(name), "wb") as f:
        pkl.dump(metalabel, f)

    print("Complete!")

def load_metadata(name=""):
    metadata = []
    metalabel = []
    with open("data/metadata/metadata{}_train.pkl".format(name), "rb") as f:
        metadata = pkl.load(f)

    with open("data/metadata/metalabel{}_train.pkl".format(name), "rb") as f:
        metalabel = pkl.load(f)

    return metadata, metalabel