from src.utils.dataloader import load_dataset
from src.models.CNN import CNN
from src.models.SCNN import SCNN

import torch
import torch.nn as nn
import torch.optim as optim

from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils

import os
import json
import pandas as pd

DEVICE = 'cpu'

def get_DEVICE():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

# DEVICE = 'cpu'

superclasses = [
    ['BRCA', 'KICH', 'KIRC', 'LUAD', 'LUSC', 'MESO', 'SARC', 'UCEC'],
    ['BLCA', 'CESC', 'HNSC', 'KIRP', 'PAAD', 'READ', 'STAD'],
    ['DLBC', 'LGG', 'PRAD', 'TGCT', 'THYM', 'UCS'],
    ['ACC', 'CHOL', 'LIHC'],
    ['ESCA', 'PCPG', 'SKCM', 'THCA', 'UVM']
]

import snntorch as snn


def load_params(typeof='cnn', num_class=0):
    '''
    Load the parameters of the model.

    Args:
        typeof (str): type of the model, cnn or scnn.
        num_class (int): number of classes.

    Returns:
        params (dict): dictionary with the parameters of the model.
    '''
    assert typeof.startswith(('cnn', 'scnn')), 'Type of model not supported'
    params_path = os.path.join('data', 'params', typeof, f'{typeof}_class{num_class}.json')

    return json.loads(open(params_path).read())

def set_model(model_name: str, num_classes: int, params: dict, neurons_per_classes=25):
    '''
    Set the model with its parameters, given the name of the model, the class number and the parameters.

    Args:
        model_name (str): name of the model.
        num_class (int): class index.
        params (dict): dictionary with the parameters of the model.
    
    Returns:
        model (nn.Module): model with the parameters set.
    '''
    filter_numbers = [params['nf1'], params['nf2'], params['nf3']]
    convolution_windows = [params['cw1'], params['cw2'], params['cw3']]
    max_pooling_windows = [params['pw1'], params['pw2'], params['pw3']]
    if model_name.startswith('cnn'):
        dropout = [params['dropout_0'], params['dropout_1']]
    else:
        beta = params['beta']
    final_nf = params['nf4']
    
    if model_name.startswith('cnn'):
        model = CNN(num_classes, filter_numbers, convolution_windows, max_pooling_windows, final_nf, dropout)
    else:
        grad = snn.surrogate.fast_sigmoid()
        pop_outputs = num_classes * neurons_per_classes

        model = model = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=filter_numbers[0], kernel_size=convolution_windows[0]),
        nn.MaxPool1d(kernel_size=max_pooling_windows[0]),
        snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True, learn_beta=True),
        nn.Conv1d(in_channels=filter_numbers[0], out_channels=filter_numbers[1], kernel_size=convolution_windows[1]),
        nn.MaxPool1d(kernel_size=max_pooling_windows[1]),
        snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True, learn_beta=True),
        nn.Conv1d(in_channels=filter_numbers[1], out_channels=filter_numbers[2], kernel_size=convolution_windows[2]),
        nn.MaxPool1d(kernel_size=max_pooling_windows[2]),
        snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True, learn_beta=True),
        nn.Flatten(),
        nn.LazyLinear(final_nf),
        nn.Linear(final_nf, pop_outputs),
        snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True, output=True, learn_beta=True)
    )
    model.to(DEVICE)
    return model

def load_model(model_path: str, typeof: str, num_class: int, params: dict):
    '''
    Load the model given the path, the type and the class number.

    Args:
        model_path (str): path of the model.
        typeof (str): type of the model.
        num_class (int): class index.

    Returns:
        model (nn.Module): model loaded.
    '''
    assert typeof.startswith(('cnn', 'scnn')), 'Type of model not supported'
    filepath = os.path.join(model_path, f'{typeof}_class{num_class}.pth')
    model = set_model(typeof, num_class, params)
    model.load_state_dict(torch.load(filepath))
    return model

from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)
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
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return correct

def test_accuracy(data_loader, net, num_steps, population_code=False, num_classes=False):
  with torch.no_grad():
    total = 0
    acc = 0
    net.eval()

    data_loader = iter(data_loader)
    for data, targets in data_loader:
      data = data.to(DEVICE)
      targets = targets.to(DEVICE)
      utils.reset(net)
      spk_rec, _ = net(data)

      if population_code:
        acc += SF.accuracy_rate(spk_rec.unsqueeze(0), targets, population_code=True, num_classes=num_classes) * spk_rec.size(1)
      else:
        acc += SF.accuracy_rate(spk_rec.unsqueeze(0), targets) * spk_rec.size(1)
        
      total += spk_rec.size(1)

  return acc/total

def experiment(model_name: str, params: dict, metalabel: int, labels_of_metaclass, epochs:int, mode: str = 'train', neurons_per_classes=25):
    num_classes = len(labels_of_metaclass)

    print(f'METACLASS LABELS: {labels_of_metaclass}')
    train_dataloader, test_dataloader = load_dataset(name='cancer', batch_size=params['batch_size'],
                                                     metalabel=metalabel, labels_of_metaclass=labels_of_metaclass)

    if mode == 'train':
        model = set_model(model_name, num_classes, params, neurons_per_classes)
    else:
        model = load_model(os.path.join('models', model_name), model_name, metalabel, params)
    epochs = epochs  # TODO: be careful
    
    
    loss_fn = SF.mse_count_loss(correct_rate=1.0, incorrect_rate=0.0, population_code=True, num_classes=num_classes) if model_name.startswith('scnn') else nn.CrossEntropyLoss()
    
    # optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'])
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], betas=(0.9, 0.999))
    num_steps = params['num_step']

    for t in range(epochs):
        if t % 10 == 0:
            print(f"Epoch {t + 1}\n-------------------------------")
        if mode == 'train':
            if model_name.startswith('cnn'):
                train(train_dataloader, model, loss_fn, optimizer)
            else:
                 backprop.BPTT(model, train_dataloader, num_steps=num_steps,
                            optimizer=optimizer, criterion=loss_fn, time_var=False, device=DEVICE)

        if t % 10 == 0:
            if model_name.startswith('cnn'):
                accuracy = test(test_dataloader, model, loss_fn)
            else:
                accuracy = test_accuracy(test_dataloader, model, num_steps, population_code=True, num_classes=num_classes)*100
            print(accuracy)
    print("Done!")
    print(f"Accuracy for superclass {metalabel}: {accuracy:.3f}")

    
    # input('Premi un tasto per concludere l esperimento...')
    # Sorre: Ho modificato il path per salvare i modelli all'interno della cartella models e non data/models
    save_filepath = os.path.join('models', model_name)
    if not os.path.exists(save_filepath):
        os.makedirs(save_filepath)

    if mode == 'train':    
        torch.save(model.state_dict(), os.path.join(save_filepath, f'{model_name}_class{metalabel}.pth'))

    return round(accuracy, 3)
    # Call the save_model function to save the model

if __name__ == '__main__':
    model_name = 'scnn'
    epochs = 100
    DEVICE = get_DEVICE()

    df = pd.DataFrame(columns=['metalabel', 'neurons_per_classes', 'accuracy'])

    if not os.path.exists(os.path.join('data', 'results')):
        os.makedirs(os.path.join('data', 'results'))
    
    for neurons_per_classes in [25, 50, 100]:
        for metalabel in range(len(superclasses)):
            params = load_params(model_name, metalabel)
            metaclass_labels = superclasses[metalabel]
            acc = experiment(
                model_name=model_name,
                params=params,
                metalabel= metalabel,
                labels_of_metaclass=metaclass_labels,
                epochs=epochs,
                neurons_per_classes=neurons_per_classes
            )
            # df = df.append({'metalabel': metalabel, 'neurons_per_classes': neurons_per_classes, 'accuracy': acc}, ignore_index=True)
            df.loc[len(df)] = [metalabel, neurons_per_classes, acc]
    df.to_csv(os.path.join('data', 'results', f'{model_name}_population_encoding.csv'))
