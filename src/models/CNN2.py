import torch
import torch.nn as nn

class CNN2(nn.Module):
    def __init__(
            self,
            num_classes,
            filter_numbers, 
            convolution_windows, 
            max_pooling_windows, 
            final_nf,
            dropout
        ):
        super(CNN2, self).__init__()
        self.filter_numbers = filter_numbers
        self.convolution_windows = convolution_windows
        self.max_pooling_windows = max_pooling_windows
        self.final_nf = final_nf
        self.dropout = dropout
        # First Convolutional Layer
        self.conv1d_1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.filter_numbers[0], kernel_size=self.convolution_windows[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.max_pooling_windows[0]))
        # Second Convolutional Layer
        self.conv1d_2 = nn.Sequential(
            nn.Conv1d(in_channels=self.filter_numbers[0], out_channels=self.filter_numbers[1],
                      kernel_size=self.convolution_windows[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.max_pooling_windows[1]))
        
        # Third Convolutional Layer
        self.conv1d_3 = nn.Sequential(
            nn.Conv1d(in_channels=self.filter_numbers[1], out_channels=self.filter_numbers[2],
                      kernel_size=self.convolution_windows[2]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.max_pooling_windows[2]))
        
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Dropout(self.dropout[0]),
            nn.Linear(self.self.final_nf),
            nn.Dropout(self.dropout[1]),
            nn.Linear(self.final_nf, num_classes)
        )

    def forward(self, x):
        #x = torch.unsqueeze(x, 1) This is already done in the dataset [Check the data_loading_functions.py]
        out = self.conv1d_1(x)
        out = self.conv1d_2(out)
        out = self.conv1d_3(out)
        out = out.view(out.size(0), -1)  # It flattens the output
        out = self.fc(out)
        return out