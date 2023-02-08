import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, channels, stride, kernel_size=4, global_pool=None, convpool=None, batchnorm=False):
        super().__init__()

        model = []

        nr_layer = len(channels) - 1
        padding = (kernel_size - 1) // 2

        for i in range(nr_layer):
            model.append(nn.Linear(channels[i], channels[i+1]))
            model.append(nn.ReLU())
        self.model = nn.Sequential(*model)


    def en_net_layer(self, in_channel, out_channel, kernel_size, dropout, stride, padding):
        model = []
        model.append(nn.BatchNorm2d(in_channel))
        model.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding))
        model.append(nn.ReLU(inplace=True))
        model.append(nn.BatchNorm2d(out_channel))
        model.append(nn.Dropout(dropout, inplace=True))
        return model
    

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
