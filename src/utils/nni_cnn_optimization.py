import matplotlib.pyplot as plt
import numpy as np

from src.utils.utils import *
from src.utils.data_loading_functions import *
from src.utils.metadata_functions import *
from src.utils.statistics import *
from src.utils.superclasses_functions import *

# Load data
label_path = "data/MLinApp_course_data/tcga_mir_label.csv"
data_path = "data/MLinApp_course_data/tcga_mir_rpm.csv"
miRNA_data, miRNA_labels, miRNA_tissues = load_data(data_path, label_path)

# Adjust data
miRNA_data, miRNA_labels, miRNA_tissues, labels, dictionary, lab = class_balancing(miRNA_data, miRNA_labels, miRNA_tissues)
# Z-Score normalization
miRNA_data = normalize_data(miRNA_data)
# Splitting the data
train_data, val_data, train_label, val_label = split_data(miRNA_data, miRNA_labels)
print('hello')

