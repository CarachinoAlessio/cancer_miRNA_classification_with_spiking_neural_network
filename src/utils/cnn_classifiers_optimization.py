'''
TODO: FILE DA ELIMINARE
import os, platform
import pandas as pd
import numpy as np
import nni
from torchvision.datasets import MNIST

from src.models.cnn_classifier import CNN
from src.utils.feature_selection import FeatureSelection
import torch
from torch import nn
import signal
from nni.experiment import Experiment
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms

def load_dataset(name='', batch_size=128, shuffle_train=True, shuffle_test=False, val=False):
    """
        :param name: name of the dataset
        :param batch_size: batch size (default 128)

        Available datasets:
        - "MNIST"
        - "GTSRB"
        - "SVHN"
    """
    root_datasets = os.path.dirname(torchvision.datasets.__file__)

    if name == 'MNIST':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
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
            shuffle=True,
            num_workers=2
        )

        test_dataset = torch.utils.data.DataLoader(
            training_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )

        return train_dataset, test_dataset

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return correct


experiment = Experiment('local')

search_space = {
    "nf1": {"_type": "randint", "_value": [2, 256]},
    "nf2": {"_type": "randint", "_value": [2, 256]},
    "nf3": {"_type": "randint", "_value": [2, 256]},
    "nf4": {"_type": "randint", "_value": [2, 256]},
    "cw1": {"_type": "randint", "_value": [4, 64]},
    "cw2": {"_type": "randint", "_value": [4, 64]},
    "cw3": {"_type": "randint", "_value": [4, 64]},
    "pw1": {"_type": "randint", "_value": [4, 64]},
    "pw2": {"_type": "randint", "_value": [4, 64]},
    "pw3": {"_type": "randint", "_value": [4, 64]},

    "batch_size": {"_type": "choice", "_value": [32, 64, 128]},
    "lr": {"_type": "uniform", "_value": [0.0001, 0.1]}
}


cnn_file_path = os.path.join("src", "models", "cnn_classifier.py")
experiment.config.trial_command = f'python {cnn_file_path}'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'Anneal'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.max_trial_number = 10
experiment.config.trial_concurrency = 2

device = 'cpu'


if __name__ == '__main__':

    superclasses = [
        ['easp', 'prova'],
        ['fppp', 'ciao']
    ]

    data = np.asarray([
        [10, 2, 4, 5, 'easp'],
        [10, 2, 4, 5, 'easp'],
        [1, 2, 100, 50000, 'prova'],
        [10, 2, 3, 50000, 'prova'],
        [3, 4, 4, 5, 'fppp'],
        [9, 9, 4, 5, 'ciao']
    ])

    base_path = os.path.join("data", "features")


    assert os.path.isfile(os.path.join(base_path, 'selected_features.csv'))
    df = pd.read_csv(os.path.join(base_path, 'selected_features.csv'))

    fs = FeatureSelection('data/features/dummy_features.csv', superclasses)

    # reduced_data, labels = fs.reduce_dimensionality(data)

    for s in range(len(superclasses)):
        params = {
            "nf1": 2,
            "nf2": 2,
            "nf3": 2,
            "nf4": 4,
            "cw1": 4,
            "cw2": 4,
            "cw3": 4,
            "pw1": 4,
            "pw2": 4,
            "pw3": 4,
            "batch_size": 32,
            "lr": 0.01
        }

        optimized_params = nni.get_next_parameter()
        params.update(optimized_params)
        print(params)

        # data_i_superclass, labels_i_superclass = [(i, j) for i, j in zip(reduced_data, labels) if j in superclasses[s]]
        num_classes = 10
        filter_numbers = [params['nf1'], params['nf2'], params['nf3']]
        convolution_windows = [params['cw1'], params['cw2'], params['cw3']]
        max_pooling_windows = [params['pw1'], params['pw3'], params['pw3']]
        final_nf = params['nf4']

        train_dataloader, test_dataloader = load_dataset(name='MNIST', batch_size=params['batch_size'])


        model = CNN(num_classes, filter_numbers, convolution_windows, max_pooling_windows, final_nf).to(device)
        epochs = 2
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'])

        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
            accuracy = test(test_dataloader, model, loss_fn)
            print(accuracy)
            nni.report_intermediate_result(accuracy)
        nni.report_final_result(accuracy)

        experiment.run(8080)
        input('Enter a key to stop the experiment...')
        # match platform.system():
        #     case 'Windows':
        #         os.system('pause')
        #     case '_':
        #         signal.pause()
        #

        experiment.stop()
'''