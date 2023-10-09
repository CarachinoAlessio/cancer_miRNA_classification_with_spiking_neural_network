import pprint
import numpy as np
import csv


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
            lbl = lbl.replace("TCGA-", "")

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
