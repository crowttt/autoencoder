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
            layer = self.en_net_layer(
                in_channel=channels[i],
                out_channel=channels[i+1],
                kernel_size=kernel_size,
                dropout=0.5,
                stride=stride[i],
                padding=padding
            )
            model = model + layer
        self.model = nn.Sequential(*model)


    def en_net_layer(self, in_channel, out_channel, kernel_size, dropout, stride, padding):
        model = []
        model.append(nn.BatchNorm2d(in_channel))
        # model.append(nn.ReLU(inplace=True))
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
        padding = (kernel_size - 1) // 2
        acti = nn.LeakyReLU(0.2)
        # acti = nn.Tanh()
        # acti = nn.ReLU(inplace=True)

        nr_layer = len(channels) - 1
        for i in range(nr_layer):
            model.append(nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=kernel_size, stride=stride[i], padding=padding))
            model.append(nn.Conv2d(channels[i + 1], channels[i + 1], kernel_size=kernel_size, stride=1, padding=padding))
            if i == 0 or i == 1:
                model.append(nn.Dropout(p=0.2))
            if not i == len(channels) - 2:
                model.append(acti)
        model.append(nn.ReflectionPad2d(1))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
