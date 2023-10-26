import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt


class SCNN(nn.Module):
    def __init__(self, num_classes, filter_numbers, convolution_windows, max_pooling_windows, final_nf, beta):
        super(SCNN, self).__init__()
        self.filter_numbers = filter_numbers
        self.convolution_windows = convolution_windows
        self.max_pooling_windows = max_pooling_windows
        self.final_nf = final_nf
        self.beta = beta
        self.spike_grad = surrogate.fast_sigmoid()
        # First Convolutional Layer
        self.conv1d_1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.filter_numbers[0], kernel_size=self.convolution_windows[0]),
            nn.MaxPool1d(kernel_size=self.max_pooling_windows[0]),
            snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, init_hidden=True))
            
        # Second Convolutional Layer
        self.conv1d_2 = nn.Sequential(
            nn.Conv1d(in_channels=self.filter_numbers[0], out_channels=self.filter_numbers[1],
                      kernel_size=self.convolution_windows[1]),
            nn.MaxPool1d(kernel_size=self.max_pooling_windows[1]),
            snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, init_hidden=True))
        # Third Convolutional Layer
        self.conv1d_3 = nn.Sequential(
            nn.Conv1d(in_channels=self.filter_numbers[1], out_channels=self.filter_numbers[2],
                      kernel_size=self.convolution_windows[2]),
            nn.MaxPool1d(kernel_size=self.max_pooling_windows[2]),
            snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, init_hidden=True))
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(self.final_nf),
            nn.Linear(self.final_nf, num_classes),
            snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, init_hidden=True, output=True)
        )

    def forward(self, num_steps, data):
        mem_rec = []
        spk_rec = []
        utils.reset(self.conv1d_1)  # resets hidden states for all LIF neurons in net
        utils.reset(self.conv1d_2)
        utils.reset(self.conv1d_3)
        utils.reset(self.fc)
        data = torch.unsqueeze(data, 1)

        for step in range(num_steps):
            spk_out = self.conv1d_1(data)
            spk_out = self.conv1d_2(spk_out)
            spk_out = self.conv1d_3(spk_out)
            spk_out, mem_out = self.fc(spk_out)

            spk_rec.append(spk_out)
            mem_rec.append(mem_out)
  
        return torch.stack(spk_rec), torch.stack(mem_rec)