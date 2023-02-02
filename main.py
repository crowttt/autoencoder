import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import numpy as np
import torchvision.utils as vutils
import argparse
from tqdm import tqdm

from model import AutoEncoder
from config import Config
from dataset import ntu_skeleton


def autoencoder_loss(reconstructed_x, original_x):
    # reconstruction_loss = torch.nn.BCELoss(reduction='sum')
    reconstruction_loss = nn.MSELoss()
    return reconstruction_loss(reconstructed_x, original_x)


def load_optimizer(config, model):
    if config.optimizer == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.base_lr,
            momentum=0.9,
            nesterov=config.nesterov,
            weight_decay=config.weight_decay)
    elif config.optimizer == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.base_lr,
            weight_decay=config.weight_decay)
    else:
        raise ValueError()
    return optimizer


# def worker_init_fn():
#     np.random.seed(np.random.get_state())
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--gpu_ids', type=str, default=0, required=False, help="specify gpu ids")
    parser.add_argument('--gpu_ids', nargs='+', help='specify gpu ids', default='0')
    parser.add_argument('--dataset', type=str, default="ntu_skeleton", help="specify dataset")
    parser.add_argument('--data_path', default="", required=True)
    parser.add_argument('--batchnorm', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=2048)


    args = parser.parse_args()
    config = Config(args)

    # create the network
    net = AutoEncoder(config)
    data = ntu_skeleton(config.data_path)
    optimizer = load_optimizer(config, net)

    data_loader = DataLoader(data, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers,
                              worker_init_fn=worker_init_fn, pin_memory=True)

    if len(config.gpu_ids) > 1:
        net = nn.DataParallel(net, device_ids=config.gpu_ids)
    
    if config.device == 'cuda':
        net.to(config.device)


    for _ in range(config.epochs):
        epoch_val_loss = []

        # begin iteration
        pbar = tqdm(data_loader)
        for b, data_input in enumerate(pbar):
            data_input = data_input.float()
            if config.device == 'cuda':
                data_input = data_input.to(config.device, dtype=torch.float)

            output = net(data_input)

            loss = autoencoder_loss(output, data_input)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    main()