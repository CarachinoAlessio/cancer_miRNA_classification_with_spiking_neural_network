import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, GTSRB, SVHN
from torch.utils.data import Subset

from src.utils.data_loading_functions import CancerDataset
from sklearn.model_selection import train_test_split

import os

DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(name='', batch_size=128, shuffle_train=True, shuffle_test=False, val=False, metalabel=0, labels_of_metaclass=[]):
    """
        :param name: name of the dataset
        :param batch_size: batch size (default 128)

        Available datasets:
        - "MNIST"
        - "GTSRB"
    """
    root_datasets = os.path.realpath(os.path.join('data'))

    if name == 'cancer':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        training_set = CancerDataset(os.path.join('data', 'features', f'reduced_data_metalabel{metalabel}_train.csv'), labels_of_metaclass)
        test_set = CancerDataset(os.path.join('data', 'features', f'reduced_data_metalabel{metalabel}_val.csv'), labels_of_metaclass)

        train_dataloader = torch.utils.data.DataLoader(
            training_set,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=1,
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=1,
        )

        return train_dataloader, test_dataloader
