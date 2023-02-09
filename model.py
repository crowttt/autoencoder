import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, channels, stride, kernel_size=4):
        super().__init__()

        model = []

        nr_layer = len(channels) - 1
        padding = (kernel_size - 1) // 2

        for i in range(nr_layer):
            model.append(nn.Linear(channels[i], channels[i+1]))
            model.append(nn.ReLU())
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x


class Decoder(nn.Module):
    def __init__(self, channels, stride, kernel_size=3):
        super().__init__()

        model = []

        nr_layer = len(channels) - 1
        for i in range(nr_layer):
            model.append(nn.Linear(channels[i], channels[i+1]))
            model.append(nn.ReLU())

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
