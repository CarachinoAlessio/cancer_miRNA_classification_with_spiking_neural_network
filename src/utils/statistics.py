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


def plot_confusion_matrix(actual_classes : np.array, predicted_classes : np.array, sorted_labels : list):
    #Forse non utilizzata
    matrix = confusion_matrix(actual_classes, predicted_classes, labels=lab2int(sorted_labels))
    
    plt.figure(figsize=(20,20))
    sns.heatmap(matrix, annot=True, xticklabels=sorted_labels, yticklabels=sorted_labels, cmap="Blues", fmt="g")
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')

    plt.show()

def show_matrix(actual_classes=None, predicted_classes=None, labels=None,cm=None):
    np.set_printoptions(precision=2)
    if cm is None:
        if actual_classes is None or predicted_classes is None:
            print("Actual or predicted classes not defined")
            return
        cm = confusion_matrix(actual_classes, predicted_classes,normalize='true')
    if labels is None:
        print("Labels not defined")
        return
    cmp = ConfusionMatrixDisplay(cm, display_labels=labels)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            cm[i,j] = format(cm[i, j], '.2f')
    fig, ax = plt.subplots(figsize=(20,20))
    cmp.plot(ax=ax)
    return cm
