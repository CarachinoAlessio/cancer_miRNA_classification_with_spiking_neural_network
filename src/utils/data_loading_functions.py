#Data loading functions
## Handling data, dataset, normalize data, remove classes
##These function are used for the classic ML part, for the CNN use the pytorch Dataloader

import matplotlib.pyplot as plt
import numpy as np
from src.utils.utils import *
from sklearn import datasets, svm
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import scipy

def load_data(data_path:str, label_path:str):
        miRna_label, miRna_tissues = extract_label(label_path)
        miRna_data = np.genfromtxt(data_path, delimiter=',')[1:,0:-1]
        # Deleting all the data that came from TARGET-* labels
        number_to_delete = abs(len(miRna_label) - miRna_data.shape[0])
        miRna_data = miRna_data[number_to_delete:,:]
        return miRna_label,miRna_data,miRna_tissues
    
def class_balancing(miRna_label,miRna_data,miRna_tissues):
    # Using only the 29 classes to make it comparable with the original paper

    # not used classes: 'COAD' 'LAML' 'OV' 'GBM'
    print("Adjusting dataset...")
    to_delete = []
    for i in range(len(miRna_label)):
        if miRna_label[i] == 'GBM' or miRna_label[i] == 'COAD' or miRna_label[i] == 'LAML' or miRna_label[i] == 'OV':
            to_delete.append(i)

    #Remove GBM data from miRna_data and miRna_label
    miRna_data = np.delete(miRna_data, to_delete, axis=0)
    miRna_label = np.delete(miRna_label, to_delete, axis=0)
    miRna_tissues = np.delete(miRna_tissues, to_delete, axis=0)
    print("Removed classes: 'COAD' 'LAML' 'OV' 'GBM'!")
    print("\n")
    print("Balancing BRCA data...")
    #Balance BRCA data
    index_BRCA = []
    for i in range(len(miRna_label)):
        if miRna_label[i] == 'BRCA':
            index_BRCA.append(i)

    #set seed to make it reproducible
    np.random.seed(42)

    index_BRCA = np.random.choice(index_BRCA, 600, replace=False)
    miRna_data = np.delete(miRna_data, index_BRCA, axis=0)
    miRna_label = np.delete(miRna_label, index_BRCA, axis=0)
    miRna_tissues = np.delete(miRna_tissues, index_BRCA, axis=0)
    print("BRCA data balanced!")
    print("\n")
    print("Processing labels...")
    ## Processing labels
    label_idx, dictionary = label_processing(miRna_label)
    labels = np.unique(miRna_label, return_counts=True)
    tissues = np.unique(miRna_tissues)

    lab = []
    for i in range(len(labels[0])):
        lab.append((labels[0][i], labels[1][i], dictionary[labels[0][i]]))

    lab.sort(key= lambda x: x[1])
    print("Done!")
    print(labels[0])
    print(tissues)
    return miRna_data, miRna_label, miRna_tissues, labels, dictionary, lab

def normalize_data(miRna_data):
    # Z-score normalization
    print("Normalizing data...")
    miRna_data = scipy.stats.zscore(miRna_data, axis=1)
    assert np.isnan(miRna_data).sum() == 0
    print("Data normalized")
    return miRna_data

def split_data(miRna_data,miRna_label):
    print("Splitting data...")
    train_data, val_data, train_label, val_label = train_test_split(miRna_data, miRna_label, test_size=0.20, random_state=42)
    n_classes = np.unique(train_label).size
    print("There are", n_classes, " classes")
    print("Training set dimensions: {}".format(train_data.shape))
    print("Validation set dimensions: {}".format(val_data.shape))
    #print("Test set dimensions: {}".format(eva_data.shape))
    print("\n")
    print("Dimensions of a single sample: {}".format(train_data[0].shape))
    return train_data, val_data, train_label, val_label