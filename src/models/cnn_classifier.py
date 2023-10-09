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
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(in_channels=300, out_channels=filters_number[0], kernel_size=convolution_windows[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=max_pooling_windows[0]))
        # Secondo strato convoluzionale
        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=filters_number[1], kernel_size=convolution_windows[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=max_pooling_windows[1]))
        # Terzo strato convoluzionale
        self.conv2d_3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=filters_number[2], kernel_size=convolution_windows[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=max_pooling_windows[2]))

        # Strato completamente connesso
        self.fc = nn.Linear(final_nf, num_classes)

    def forward(self, x):
        out = self.conv2d_1(x)
        out = self.conv2d_2(out)
        out = self.conv2d_3(out)
        out = out.view(out.size(0), -1)  # Appiattisce l'output
        out = self.fc(out)
        return out
