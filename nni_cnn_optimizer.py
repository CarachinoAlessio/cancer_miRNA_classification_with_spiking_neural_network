import torch
import torch.nn as nn
import nni
import argparse

from src.models.CNN import CNN
from src.utils.dataloader import load_dataset

device = ("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using {device}")

superclasses = [
    ['BRCA', 'KICH', 'KIRC', 'LUAD', 'LUSC', 'MESO', 'SARC', 'UCEC'],
    ['BLCA', 'CESC', 'HNSC', 'KIRP', 'PAAD', 'READ', 'STAD'],
    ['DLBC', 'LGG', 'PRAD', 'TGCT', 'THYM', 'UCS'],
    ['ACC', 'CHOL', 'LIHC'],
    ['ESCA', 'PCPG', 'SKCM', 'THCA', 'UVM']
]


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


def main_cnn_optimization(params, metalabel, labels_of_metaclass):
    num_classes = len(labels_of_metaclass)

    print(f'METACLASS LABELS: {labels_of_metaclass}')
    train_dataloader, test_dataloader = load_dataset(name='cancer', batch_size=params['batch_size'],
                                                     metalabel=metalabel, labels_of_metaclass=labels_of_metaclass)

    filter_numbers = [params['nf1'], params['nf2'], params['nf3']]
    convolution_windows = [params['cw1'], params['cw2'], params['cw3']]
    max_pooling_windows = [params['pw1'], params['pw2'], params['pw3']]
    dropout = [params['dropout_0'], params['dropout_1']]
    final_nf = params['nf4']

    model = CNN(num_classes, filter_numbers, convolution_windows, max_pooling_windows, final_nf, dropout)
    model.to(device)
    epochs = 100  # TODO: be careful
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'])
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        accuracy = test(test_dataloader, model, loss_fn)
        print(accuracy)
        nni.report_intermediate_result(accuracy)
    nni.report_final_result(accuracy)
    # input('Premi un tasto per concludere l esperimento...')


def get_params():
    ''' Get parameters from command line '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--nf1", type=int, default=2)
    parser.add_argument("--nf2", type=int, default=2)
    parser.add_argument("--nf3", type=int, default=2)
    parser.add_argument("--nf4", type=int, default=2)
    parser.add_argument("--cw1", type=int, default=4)
    parser.add_argument("--cw2", type=int, default=4)
    parser.add_argument("--cw3", type=int, default=4)
    parser.add_argument("--pw1", type=int, default=4)
    parser.add_argument("--pw2", type=int, default=4)
    parser.add_argument("--pw3", type=int, default=4)
    parser.add_argument("--dropout_0", type=float, default=.5)
    parser.add_argument("--dropout_1", type=float, default=.5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)

    parser.add_argument("--superclass", type=int, help='Please specify the superclass you want to run the experiment '
                                                       'for.')
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    superclass_target = 2
    metaclass_labels = superclasses[superclass_target]
    print(f'RUNNING EXPERIMENT FOR SUPERCLASS {superclass_target}')

    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        # logger.debug(tuner_params)
        params = vars(get_params())
        params.update(tuner_params)
        main_cnn_optimization(params, superclass_target, metaclass_labels)
    except Exception as exception:
        # logger.exception(exception)
        raise