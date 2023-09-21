import numpy as np
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
