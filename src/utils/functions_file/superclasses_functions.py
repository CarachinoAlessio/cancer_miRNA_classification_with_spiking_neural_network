import numpy as np
import copy as cp
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import datasets, svm
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
from typing import Tuple
from sklearn.metrics import confusion_matrix
import itertools
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt

def confusion_matrix_to_metric_matrix(self,n_classes,labels):
        #Convert the confusion matrix into a metric matrix (superclass identification)
        metric_matrix = np.zeros((n_classes,n_classes))

        for i in range(n_classes):
            for j in range(n_classes):
                if i == j:
                    metric_matrix[i,j] = 0 #cm[i,j]
                else:
                    metric_matrix[i,j] = 1/(cm[i,j] + cm[j,i]) if cm[i,j] != 0 or cm[j,i] != 0 else 100 #1/cm[i,j] if cm[i,j] != 0 else 100 #cm[i,j] + cm[j,i]
        #make it simetric
        #for i in range(n_classes):
        #    for j in range(n_classes):
        #        if i != j:
        #            metric_matrix[j,i] = metric_matrix[i,j]
        # normalize every element of the matrix in the range [0,1]
        metric_matrix = metric_matrix/np.max(metric_matrix)




        #Plotting the metric matrix
        cmp = ConfusionMatrixDisplay(metric_matrix, display_labels=labels[0])
        for i, j in itertools.product(range(metric_matrix.shape[0]), range(metric_matrix.shape[1])):
                metric_matrix[i,j] = format(metric_matrix[i, j], '.2f')
        fig, ax = plt.subplots(figsize=(20,20))
        cmp.plot(ax=ax)
        return metric_matrix

def create_dendogram(self,metric_matrix,labels):
        #We know that i is the class to predict and j is the predicted class. So cm[i][j] is the miss-classification ratio 
    plt.figure(figsize=(10, 7))
    plt.title("Classes dendogram")

    # Plotting dendogram

    clusters = shc.linkage(metric_matrix, 
                method='ward')
    shc.dendrogram(Z=clusters, labels=labels[0])
    plt.show()

def divide_in_superclass(self,clustering_model,labels):
    #Labels = labels[0] (ordered)
    #c_per_l = cluster per label
    c_per_l = clustering_model.labels_
    superclasses = {'0': [], '1': [], '2': [], '3': [], '4': []} #, '5': [], '6': [], '7': [] }
    for i in range(len(labels[0])):
        superclasses[str(c_per_l[i])].append(labels[0][i])
    for key in superclasses.keys():
        print(len(superclasses[key]))
        print(superclasses[key])
    print("Adjusting classes...")
    #move THCA from superclass 3 to superclass 4
    superclasses['4'].append('THCA')
    superclasses['3'].remove('THCA')
    #move UVM from superclass 0 to superclass 4
    superclasses['4'].append('UVM')
    superclasses['0'].remove('UVM')
    print("New superclasses")
    for key in superclasses.keys():
        print(len(superclasses[key]))
        print(superclasses[key])
    return superclasses
