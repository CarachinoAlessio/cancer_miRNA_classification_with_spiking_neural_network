import torch
import torch.nn as nn
import snntorch as snn
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt
import nni
import argparse
#SCNN lib
from src.models.SCNN import SCNN
from src.utils.dataloader import load_dataset

device = ("cuda" if torch.cuda.is_available() else "cpu")

superclasses = [
    ['BRCA', 'KICH', 'KIRC', 'LUAD', 'LUSC', 'MESO', 'SARC', 'UCEC'],
    ['BLCA', 'CESC', 'HNSC', 'KIRP', 'PAAD', 'READ', 'STAD'],
    ['DLBC', 'LGG', 'PRAD', 'TGCT', 'THYM', 'UCS'],
    ['ACC', 'CHOL', 'LIHC'],
    ['ESCA', 'PCPG', 'SKCM', 'THCA', 'UVM']
]

def train(dataloader, model, loss_fn, optimizer, num_steps):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        spk_rec, _ = model(num_steps, X)
        loss_val = loss_fn(spk_rec, y)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()


def test(dataloader, model, loss_fn, num_steps):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    total = 0
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            spk_rec, _ = model(num_steps, X)
            correct += SF.accuracy_rate(spk_rec, y) * spk_rec.size(1)
            total += spk_rec.size(1)
    correct /= total
    return correct

def test_accuracy(data_loader, net, num_steps, population_code=False, num_classes=False):
  with torch.no_grad():
    total = 0
    acc = 0
    net.eval()

    data_loader = iter(data_loader)
    for data, targets in data_loader:
      data = data.to(device)
      targets = targets.to(device)
      utils.reset(net)
      spk_rec, _ = net(data)

      if population_code:
        acc += SF.accuracy_rate(spk_rec.unsqueeze(0), targets, population_code=True, num_classes=num_classes) * spk_rec.size(1)
      else:
        acc += SF.accuracy_rate(spk_rec.unsqueeze(0), targets) * spk_rec.size(1)
        
      total += spk_rec.size(1)

  return acc/total


def main_scnn_optimization(params, metalabel, labels_of_metaclass):
    num_classes = len(labels_of_metaclass)

    print(f'METACLASS LABELS: {labels_of_metaclass}')
    train_dataloader, test_dataloader = load_dataset(name='cancer', batch_size=params['batch_size'],
                                                     metalabel=metalabel, labels_of_metaclass=labels_of_metaclass)

    filter_numbers = [params['nf1'], params['nf2'], params['nf3']]
    convolution_windows = [params['cw1'], params['cw2'], params['cw3']]
    max_pooling_windows = [params['pw1'], params['pw2'], params['pw3']]
    final_nf = params['nf4']
    beta = params['beta']
    num_step = params['num_step']

    #model = SCNN(num_classes, filter_numbers, convolution_windows, max_pooling_windows, final_nf, beta)#.to(device)
    model = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=filter_numbers[0], kernel_size=convolution_windows[0]),
        nn.MaxPool1d(kernel_size=max_pooling_windows[0]),
        snn.Leaky(beta=beta, spike_grad=snn.surrogate.fast_sigmoid(), init_hidden=True, learn_beta=True),
        nn.Conv1d(in_channels=filter_numbers[0], out_channels=filter_numbers[1], kernel_size=convolution_windows[1]),
        nn.MaxPool1d(kernel_size=max_pooling_windows[1]),
        snn.Leaky(beta=beta, spike_grad=snn.surrogate.fast_sigmoid(), init_hidden=True, learn_beta=True),
        nn.Conv1d(in_channels=filter_numbers[1], out_channels=filter_numbers[2], kernel_size=convolution_windows[2]),
        nn.MaxPool1d(kernel_size=max_pooling_windows[2]),
        snn.Leaky(beta=beta, spike_grad=snn.surrogate.fast_sigmoid(), init_hidden=True, learn_beta=True),
        nn.Flatten(),
        nn.LazyLinear(final_nf),
        nn.Linear(final_nf, num_classes),
        snn.Leaky(beta=beta, spike_grad=snn.surrogate.fast_sigmoid(), init_hidden=True, output=True, learn_beta=True)
    )

    model.to(device)
    epochs = 5  # TODO: be careful
    loss_fn = SF.mse_count_loss(correct_rate=1.0, incorrect_rate=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    accuracy = 0

    for epoch in range(epochs):

        avg_loss = backprop.BPTT(model, train_dataloader, num_steps=num_step,
                            optimizer=optimizer, criterion=loss_fn, time_var=False, device=device)
        accuracy = test_accuracy(test_dataloader, model, num_step)
        print(f"Epoch: {epoch}")
        print(f"Test set accuracy: {accuracy*100:.3f}%\n")
        nni.report_intermediate_result(accuracy)
    nni.report_final_result(accuracy)

    # for t in range(epochs):
    #     print(f"Epoch {t + 1}\n-------------------------------")
    #     train(train_dataloader, model, loss_fn, optimizer, num_steps=num_step)
    #     accuracy = test(test_dataloader, model, loss_fn, num_steps=num_step)
    #     print(accuracy)
    #     nni.report_intermediate_result(accuracy)
    # nni.report_final_result(accuracy)
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
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--num_step", type=int, default=50)
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
        main_scnn_optimization(params, superclass_target, metaclass_labels)
    except Exception as exception:
        # logger.exception(exception)
        raise