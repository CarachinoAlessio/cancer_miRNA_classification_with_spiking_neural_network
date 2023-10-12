import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, GTSRB, SVHN
from torch.utils.data import Subset

from src.utils.data_loading_functions import CancerDataset
from sklearn.model_selection import train_test_split

import os

DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(name='', batch_size=128, shuffle_train=True, shuffle_test=False, val=False, metalabel=0):
    """
        :param name: name of the dataset
        :param batch_size: batch size (default 128)

        Available datasets:
        - "MNIST"
        - "GTSRB"
    """
    root_datasets = os.path.realpath(os.path.join('data'))

    if name == 'MNIST':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(torch.flatten)
        ])

        # root_training = os.path.join(datasets_folder, '/MNIST/raw/train-images-idx3-ubyte')
        training_set = MNIST(
            root=root_datasets,
            train=True,
            download=True,
            transform=transform_test
        )

        test_set = MNIST(
            root=root_datasets,
            train=False,
            download=True,
            transform=transform_test
        )

        train_dataset = torch.utils.data.DataLoader(
            training_set,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=1,
        )

        test_dataset = torch.utils.data.DataLoader(
            training_set,
            batch_size=batch_size,
            shuffle=shuffle_test,
            num_workers=1,
        )
        return train_dataset, test_dataset

    elif name == 'cancer':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        training_set = CancerDataset(os.path.join('data', 'features', f'features_metalabel{metalabel}_train.csv'))
        test_set = CancerDataset(os.path.join('data', 'features', f'features_metalabel{metalabel}_val.csv'))

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

    elif name == 'GTSRB':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=(0.3400, 0.3117, 0.3209),
            #     std=(0.2717, 0.2599, 0.2658)
            #     # mean=(0.3417, 0.3126, 0.3216),
            #     # std=(0.0318, 0.0308, 0.0320)
            #     )
        ])

        training_set = GTSRB(
            root=root_datasets,
            split='train',
            download=True,
            transform=transform
        )

        test_set = GTSRB(
            root=root_datasets,
            split='test',
            download=True,
            transform=transform
        )

        if val:
            train_indexes, val_indexes = train_test_split(
                range(len(training_set)),
                test_size=0.2,
                shuffle=True,
                stratify=[y for (x, y) in training_set]
            )

            val_set = Subset(training_set, val_indexes)
            training_set = Subset(training_set, train_indexes)

        train_dataset = torch.utils.data.DataLoader(
            training_set,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=1,
        )
        if val:
            val_dataset = test_dataset = torch.utils.data.DataLoader(
                val_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2
            )
        test_dataset = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=shuffle_test,
            num_workers=1
        )

        if val:
            return train_dataset, val_dataset, test_dataset
        else:
            return train_dataset, test_dataset

    elif name == 'SVHN':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.438, 0.444, 0.473),
                (0.195, 0.198, 0.195))
        ])

        train_set = SVHN(
            root_datasets,
            split='train',
            transform=transform,
            download=True
        )

        test_set = SVHN(
            root_datasets,
            split='test',
            transform=transform,
            download=True
        )

        train_dataset = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=2,
        )

        test_dataset = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=shuffle_test,
            num_workers=2,
        )

        return train_dataset, test_dataset
