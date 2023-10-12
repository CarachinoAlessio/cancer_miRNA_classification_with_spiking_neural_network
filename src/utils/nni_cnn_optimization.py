import matplotlib.pyplot as plt
import numpy as np
import os

from src.utils.utils import *
from src.utils.data_loading_functions import *
from src.utils.metadata_functions import *
from src.utils.statistics import *
from src.utils.superclasses_functions import *
from utils.feature_selection import FeatureSelection

# Load data
label_path = os.path.join("data", "MLinApp_course_data", "tcga_mir_label.csv")
data_path = os.path.join("data", "MLinApp_course_data", "tcga_mir_rpm.csv")
miRNA_data, miRNA_labels, miRNA_tissues = load_data(data_path, label_path)

# Adjust data
miRNA_data, miRNA_labels, miRNA_tissues, labels, dictionary, lab = class_balancing(miRNA_data, miRNA_labels, miRNA_tissues)
# Z-Score normalization
miRNA_data = normalize_data(miRNA_data)
# Splitting the data
train_data, val_data, train_label, val_label = split_data(miRNA_data, miRNA_labels)
print('hello')

superclasses = [
    ['BRCA', 'KICH', 'KIRC', 'LUAD', 'LUSC', 'MESO', 'SARC', 'UCEC'],
    ['BLCA', 'CESC', 'HNSC', 'KIRP', 'PAAD', 'READ', 'STAD'],
    ['DLBC', 'LGG', 'PRAD', 'TGCT', 'THYM', 'UCS'],
    ['ACC', 'CHOL', 'LIHC'],
    ['ESCA', 'PCPG', 'SKCM', 'THCA', 'UVM']
]

data = np.vstack(train_data, train_label)

fs = FeatureSelection('data/features/save_features.csv', superclasses)

selected_features_idx = fs.find_features_subset(data, method='fisher', n_features=300, export_to_csv=True)

reduced_data, labels = fs.reduce_dimensionality(data)

print('stop')