import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
from tqdm import tqdm

# from model import AutoEncoder
from model import Encoder, Decoder
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
    parser.add_argument('--gpu_ids', nargs='+', help='specify gpu ids', default='0')
    parser.add_argument('--dataset', type=str, default="ntu_skeleton", help="specify dataset")
    parser.add_argument('--data_path', default="", required=True)
    parser.add_argument('--batchnorm', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=2048)


    args = parser.parse_args()
    config = Config(args)

    # # info
    if torch.cuda.is_available():
        print("GPU: ", torch.cuda.current_device())
    else:
        sys.exit("I want a cup of milk tea. Thanks!")

    # create the network
    min_loss = sys.float_info.max
    encode = Encoder(config.en_channels, config.en_stride, kernel_size=config.kernel)
    decode = Decoder(config.de_channels, config.de_stride, kernel_size=config.kernel)
    encode_optimizer = load_optimizer(config, encode)
    decode_optimizer = load_optimizer(config, decode)

    # Dataset
    data = ntu_skeleton(config.data_path)
    data_loader = DataLoader(data, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers,
                              worker_init_fn=worker_init_fn, pin_memory=True)


    if len(config.gpu_ids) > 1:
        encode = nn.DataParallel(encode, device_ids=config.gpu_ids)
        decode = nn.DataParallel(decode, device_ids=config.gpu_ids)
    
    if torch.cuda.is_available():
        encode.to(config.device)
        decode.to(config.device)


    for i in range(config.epochs):
        epoch_val_loss = []
        total_loss = 0.0
        encode.train()
        decode.train()

        # begin iteration
        pbar = tqdm(data_loader)
        for b, data_input in enumerate(pbar):
            data_input = data_input.float()
            if torch.cuda.is_available():
                data_input = data_input.to(config.device, dtype=torch.float)

            embed = encode(data_input)
            output = decode(embed)

            loss = autoencoder_loss(output.cuda(), data_input.cuda())

            encode_optimizer.zero_grad()
            decode_optimizer.zero_grad()
            loss.backward()
            encode_optimizer.step()
            decode_optimizer.step()
            total_loss += loss.item()

        if total_loss / len(data_loader) <= min_loss:
            min_loss = total_loss / len(data_loader)
        
            torch.save({ 
                'model_state_dict': encode.state_dict(), 
                'optimizer_state_dict': encode_optimizer.state_dict()}, 'saved_model/encoder.pt')
        print("Training: ", i, " traing loss: ", total_loss / len(data_loader))
        print("Min loss: ",min_loss)

if __name__ == '__main__':
    main()