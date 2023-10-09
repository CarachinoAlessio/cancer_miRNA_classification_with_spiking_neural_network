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

class ModelSelection:
    def __init__(self) -> None:
        pass

    def cross_val_predict(self,model, kfold : KFold, X : np.array, y : np.array) -> Tuple[np.array, np.array, np.array]:

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
    
    def lab2int(self,labels,dictionary):
        lab = []
        for l in labels:
            lab.append(dictionary[l])
        return lab
    
    def plot_confusion_matrix(self,actual_classes : np.array, predicted_classes : np.array, sorted_labels : list):

        matrix = confusion_matrix(actual_classes, predicted_classes, labels=self.lab2int(sorted_labels))
        
        plt.figure(figsize=(20,20))
        sns.heatmap(matrix, annot=True, xticklabels=sorted_labels, yticklabels=sorted_labels, cmap="Blues", fmt="g")
        plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')

        plt.show()

    def show_matrix(self,actual_classes, predicted_classes,labels):
        np.set_printoptions(precision=2)
        cm = confusion_matrix(actual_classes, predicted_classes,normalize='true')
        cmp = ConfusionMatrixDisplay(cm, display_labels=labels[0])
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                cm[i,j] = format(cm[i, j], '.2f')
        fig, ax = plt.subplots(figsize=(20,20))
        cmp.plot(ax=ax)