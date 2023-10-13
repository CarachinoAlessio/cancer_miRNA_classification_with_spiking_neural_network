import torch
import torch.nn as nn
import nni
import argparse

from src.utils.dataloader import load_dataset

device = 'cpu'
# superclasses = [
#     ['BRCA', 'KICH', 'KIRC', 'LUAD', 'LUSC', 'MESO', 'SARC', 'UCEC'],
#     ['BLCA', 'CESC', 'HNSC', 'KIRP', 'PAAD', 'READ', 'STAD'],
#     ['DLBC', 'LGG', 'PRAD', 'TGCT', 'THYM', 'UCS'],
#     ['ACC', 'CHOL', 'LIHC'],
#     ['ESCA', 'PCPG', 'SKCM', 'THCA', 'UVM']
# ]


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

    # Definire la classe per la CNN
    class CNN(nn.Module):
        def __init__(self, num_classes):
            super(CNN, self).__init__()
            self.filter_numbers = [params['nf1'], params['nf2'], params['nf3']]
            self.convolution_windows = [params['cw1'], params['cw2'], params['cw3']]
            self.max_pooling_windows = [params['pw1'], params['pw2'], params['pw3']]
            self.final_nf = params['nf4']
            # Primo strato convoluzionale
            self.conv1d_1 = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=self.filter_numbers[0], kernel_size=self.convolution_windows[0]),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=self.max_pooling_windows[0]))
            # Secondo strato convoluzionale
            self.conv1d_2 = nn.Sequential(
                nn.Conv1d(in_channels=self.filter_numbers[0], out_channels=self.filter_numbers[1],
                          kernel_size=self.convolution_windows[1]),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=self.max_pooling_windows[1]))
            # Terzo strato convoluzionale
            self.conv1d_3 = nn.Sequential(
                nn.Conv1d(in_channels=self.filter_numbers[1], out_channels=self.filter_numbers[2],
                          kernel_size=self.convolution_windows[2]),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=self.max_pooling_windows[2]))
            # Strato completamente connesso
            self.fc = nn.Sequential(
                nn.LazyLinear(self.final_nf),
                nn.Linear(self.final_nf, num_classes)
            )

        def forward(self, x):
            x = torch.unsqueeze(x, 1)
            out = self.conv1d_1(x)
            out = self.conv1d_2(out)
            out = self.conv1d_3(out)
            out = out.view(out.size(0), -1)  # Appiattisce l'output
            out = self.fc(out)
            # return torch.argmax(out, dim=1)
            return out

    model = CNN(num_classes).to(device)
    epochs = 5  # TODO: be careful
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'])

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        accuracy = test(test_dataloader, model, loss_fn)
        print(accuracy)
        nni.report_intermediate_result(accuracy)
    nni.report_final_result(accuracy)
    input('Premi un tasto per concludere l esperimento...')


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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)

    parser.add_argument("--superclass", type=int, help='Please specify the superclass you want to run the experiment '
                                                       'for.')
    args, _ = parser.parse_known_args()
    return args


# if __name__ == '__main__':
#     for i in range(len(superclasses)):
#         print(f'RUNNING EXPERIMENT FOR SUPERCLASS {i}')
#
#         try:
#             # get parameters form tuner
#             tuner_params = nni.get_next_parameter()
#             # logger.debug(tuner_params)
#             params = vars(get_params())
#             params.update(tuner_params)
#             main_cnn_optimization(params, i, superclasses[i])
#         except Exception as exception:
#             # logger.exception(exception)
#             raise
#     '''
#     try:
#         # get parameters form tuner
#         tuner_params = nni.get_next_parameter()
#         # logger.debug(tuner_params)
#         params = vars(get_params())
#         params.update(tuner_params)
#         main(params)
#     except Exception as exception:
#         # logger.exception(exception)
#         raise
#     '''
