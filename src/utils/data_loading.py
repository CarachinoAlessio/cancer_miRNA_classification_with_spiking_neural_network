#Data loading functions
## Handling data, dataset, normalize data, remove classes

import matplotlib.pyplot as plt
import numpy as np
from src.utils.utils import *
from sklearn import datasets, svm
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import scipy

class DataLoader:
    def __init__(self) -> None:
        pass

    def load_data(self,label_path:str,data_path:str):
        miRna_label, miRna_tissues = extract_label(label_path)
        miRna_data = np.genfromtxt(data_path, delimiter=',')[1:,0:-1]
        # Deleting all the data that came from TARGET-* labels
        number_to_delete = abs(len(miRna_label) - miRna_data.shape[0])
        miRna_data = miRna_data[number_to_delete:,:]
        return miRna_label,miRna_data,miRna_tissues
    
    def remove_duplicates(self,miRna_label,miRna_data,miRna_tissues):
        # Deleting all the data that came from GBM class use for cycle
        # Using only the 29 classes that are in the paper

        # not used classes: 'COAD' 'LAML' 'OV' 'GBM'
        print("Removing duplicates...")
        to_delete = []
        for i in range(len(miRna_label)):
            if miRna_label[i] == 'GBM' or miRna_label[i] == 'COAD' or miRna_label[i] == 'LAML' or miRna_label[i] == 'OV':
                to_delete.append(i)

        #Remove GBM data from miRna_data and miRna_label
        miRna_data = np.delete(miRna_data, to_delete, axis=0)
        miRna_label = np.delete(miRna_label, to_delete, axis=0)
        miRna_tissues = np.delete(miRna_tissues, to_delete, axis=0)
        print("Duplicates removed")
        print("\n")
        print("Balancing BRCA data...")
        #Balance BRCA data
        index_BRCA = []
        for i in range(len(miRna_label)):
            if miRna_label[i] == 'BRCA':
                index_BRCA.append(i)

        index_BRCA = np.random.choice(index_BRCA, 600, replace=False)
        miRna_data = np.delete(miRna_data, index_BRCA, axis=0)
        miRna_label = np.delete(miRna_label, index_BRCA, axis=0)
        miRna_tissues = np.delete(miRna_tissues, index_BRCA, axis=0)
        print("BRCA data balanced")
        print("\n")
        print("Removing the duplicates from the labels")
        ## Labeling process: removing the duplicates
        label_idx, dictionary = label_processing(miRna_label)
        labels = np.unique(miRna_label, return_counts=True)
        tissues = np.unique(miRna_tissues)

        lab = []
        for i in range(len(labels[0])):
            lab.append((labels[0][i], labels[1][i], dictionary[labels[0][i]]))

        lab.sort(key= lambda x: x[1])
        print("Results: ")
        print(labels[0])
        print(tissues)
        return miRna_label,miRna_data,miRna_tissues,dictionary
    
    def normalize_data(self,miRna_data):
        # Z-score normalization
        print("Normalizing data...")
        miRna_data = scipy.stats.zscore(miRna_data, axis=1)
        assert np.isnan(miRna_data).sum() == 0
        print("Data normalized")
        return miRna_data
    
    def split_data(self,miRna_data,miRna_label):
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
    

    