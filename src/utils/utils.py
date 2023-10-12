import pprint
import numpy as np
import csv

import copy as cp
import pickle as pkl
from sklearn.model_selection import KFold


def extract_label(file_name, verbose=False):
    data = {}
    label = []
    tissues = []
    with open(file_name, "r") as fin:
        reader = csv.reader(fin, delimiter=',')
        first = True
        for row in reader:
            lbl = row[2]
            tissue = row[13]
            if first or "TARGET" in lbl:
                first = False
                continue
            lbl = lbl.replace("TCGA-","")

            label.append(lbl)
            tissues.append(tissue)
            if lbl in data.keys():
                data[lbl] += 1
            else:
                data[lbl] = 1
    if verbose:
        print(f"Number of classes in the dataset = {len(data)}")
        pprint.pprint(data, indent=4)

    return label, tissues

def create_dictionary(labels):
    dictionary = {}
    class_names = np.unique(labels)
    for i, name in enumerate(class_names):
        dictionary[name] = i

    return dictionary

def label_processing(labels):
    new_miRna_label = []
    dictionary = create_dictionary(labels)
    for i in labels:
        new_miRna_label.append(dictionary[i])
    return new_miRna_label, dictionary

def cross_val_predict(model, kfold : KFold, X : np.array, y : np.array):

        model_ = cp.deepcopy(model)
        
        no_classes = len(np.unique(y))
        
        actual_classes = np.empty([0], dtype=int)
        predicted_classes = np.empty([0], dtype=int)
        predicted_proba = np.empty([0, no_classes]) 

        for train_ndx, test_ndx in kfold.split(X):

            train_X, train_y, test_X, test_y = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]

            actual_classes = np.append(actual_classes, test_y)

            model_.fit(train_X, train_y)
            predicted_classes = np.append(predicted_classes, model_.predict(test_X))

            try:
                predicted_proba = np.append(predicted_proba, model_.predict_proba(test_X), axis=0)
            except:
                predicted_proba = np.append(predicted_proba, np.zeros((len(test_X), no_classes), dtype=float), axis=0)

        return actual_classes, predicted_classes, predicted_proba, model_
    
def lab2int(labels,dictionary):
    lab = []
    for l in labels:
        lab.append(dictionary[l])
    return lab

def lab2super(labels, superclasses):
    for k,v in superclasses.items():
        for a in v:
            labels = list(map(lambda x: x.replace(a, k), labels))
    labels = [int(x) for x in labels]
    return labels

# Define a method to save a model using pkl, specify the path and the name of the file
def save_model(model, path, name):
    with open(path + name + '.pkl', 'wb') as fid:
        pkl.dump(model, fid)

def data_split_to_superclasses(data, superclasses):
    '''
    common function to split original data into supersets according to the superclasses
    :param data: containing samples (data + label)
    :param superclasses:
    :return:
    '''
    labels = []
    subsets = []
    for i, superclass in enumerate(superclasses):
        indici = [j for j, d in enumerate(data) if d[-1] in superclass]
        subset = data[indici]
        labels.append([s[-1] for s in subset])
        subsets.append([np.asarray(s[:-1], dtype=float) for s in subset])
    return subsets, labels
