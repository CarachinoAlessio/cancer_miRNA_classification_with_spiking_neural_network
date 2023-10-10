import torch
import torch.nn as nn
from utils.cnn_classifiers_optimization import load_dataset, train, test
import nni

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

# Definire la classe per la CNN
class CNN(nn.Module):
    def __init__(self,
                 num_classes,
                 filters_number: list[int, 3],
                 convolution_windows: list[int, 3],
                 max_pooling_windows: list[int, 3],
                 final_nf: int
                 ):
        super(CNN, self).__init__()
        # Primo strato convoluzionale
        self.conv1d_1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=filters_number[0], kernel_size=convolution_windows[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=max_pooling_windows[0]))
        # Secondo strato convoluzionale
        self.conv1d_2 = nn.Sequential(
            nn.Conv1d(in_channels=filters_number[0], out_channels=filters_number[1], kernel_size=convolution_windows[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=max_pooling_windows[1]))
        # Terzo strato convoluzionale
        self.conv1d_3 = nn.Sequential(
            nn.Conv1d(in_channels=filters_number[1], out_channels=filters_number[2], kernel_size=convolution_windows[2]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=max_pooling_windows[2]))

        # Strato completamente connesso
        self.fc = nn.Sequential(
            nn.LazyLinear(final_nf),
            nn.Linear(final_nf, num_classes)
        )

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        out = self.conv1d_1(x)
        out = self.conv1d_2(out)
        out = self.conv1d_3(out)
        out = out.view(out.size(0), -1)  # Appiattisce l'output
        out = self.fc(out)
        return out

train_dataloader, test_dataloader = load_dataset(name='MNIST', batch_size=params['batch_size'])

optimized_params = nni.get_next_parameter()
params.update(optimized_params)
print(params)

num_classes = 10
filter_numbers = [params['nf1'], params['nf2'], params['nf3']]
convolution_windows = [params['cw1'], params['cw2'], params['cw3']]
max_pooling_windows = [params['pw1'], params['pw3'], params['pw3']]
final_nf = params['nf4']

model = CNN(num_classes, filter_numbers, convolution_windows, max_pooling_windows, final_nf).to(device)
epochs = 5
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'])

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    accuracy = test(test_dataloader, model, loss_fn)
    nni.report_intermediate_result(accuracy)
nni.report_final_result(accuracy)