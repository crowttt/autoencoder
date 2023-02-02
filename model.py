import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, channels, stride, kernel_size=4, global_pool=None, convpool=None, batchnorm=False):
        super().__init__()

        model = []
        # acti = nn.LeakyReLU(0.2)

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
            # pad = (kernel_size - 1) // 2
            # # model.append(nn.ReflectionPad2d(pad))
            # model.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernel_size, stride=2))
            # if batchnorm:
            #     model.append(nn.BatchNorm2d(channels[i + 1]))
            # model.append(acti)
            # model.append(convpool(kernel_size=2, stride=2))

        # self.global_pool = global_pool

        # self.model = nn.Sequential(*model)

        # if self.compress:
        #     self.conv1x1 = nn.Conv1d(channels[-2], channels[-1], kernel_size=1)

        # self.last_conv = nn.Conv1d(channels[-1], channels[-1], kernel_size=1, bias=False)

    def en_net_layer(self, in_channel, out_channel, kernel_size, dropout, stride, padding):
        model = []
        model.append(nn.BatchNorm2d(in_channel))
        model.append(nn.ReLU(inplace=True))
        model.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding))
        model.append(nn.BatchNorm2d(out_channel))
        model.append(nn.Dropout(dropout, inplace=True))
        return model
    

    def forward(self, x):
        x = self.model(x)
        # if self.global_pool is not None:
        #     ks = x.shape[-1]
        #     x = self.global_pool(x, ks)  # F.max_pool1d
        # else:
        #     x = self.last_conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, channels, stride, kernel_size=3):
        super().__init__()

        model = []
        padding = (kernel_size - 1) // 2
        acti = nn.LeakyReLU(0.2)

        nr_layer = len(channels) - 1
        for i in range(nr_layer):
            # model.append(nn.ReflectionPad2d(pad))
            # model.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernel_size, stride=2))
            # model.append(acti)
            # model.append(convpool(kernel_size=2, stride=2))
            # if i == 0 or i == 1:
            #     model.append(nn.Dropout(p=0.2))
            # if not i == len(channels) - 2:
            #     model.append(acti)
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


class AutoEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.encode = Encoder(config.en_channels, config.en_stride, kernel_size=3, global_pool=F.avg_pool2d,
                            convpool=nn.AvgPool2d, batchnorm=config.batchnorm)
        self.decode = Decoder(config.de_channels, config.de_stride, kernel_size=3)

    def forward(self, x):
        x = self.encode(x)
        x_reco = self.decode(x)
        return x_reco
