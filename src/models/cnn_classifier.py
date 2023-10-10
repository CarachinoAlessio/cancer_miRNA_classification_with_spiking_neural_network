import torch
import torch.nn as nn


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
