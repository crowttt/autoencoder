import os
import torch
import torch.nn
import argparse
import numpy as np
from torch.utils.data import DataLoader
from dataset import ntu_skeleton
from config import Config
from model import Encoder


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--pretrained_encoder', type=str, default=None)

    args = parser.parse_args()
    config = Config()

    encoder = Encoder(config.en_channels, config.en_stride, kernel_size=config.kernel)
    pre_encoder = torch.load(args.pretrained_encoder)
    encoder.load_state_dict(pre_encoder['model_state_dict'])

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
        encoder.to(config.device)

    data = ntu_skeleton(args.data_path)
    data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=config.num_workers,
                            worker_init_fn=worker_init_fn, pin_memory=True)

    all_embed_vecs = []
    encoder.eval()
    for name, data_input in data_loader:
        data_input = data_input
        if torch.cuda.is_available():
            data_input = data_input.to(config.device, dtype=torch.float)
        min_value = data_input.min()
        std = data_input.std()
        nor_data_input = (data_input - min_value) / std
        embed_vecs = encoder(nor_data_input)

        embed_vecs = np.append(np.array(name).reshape(-1,1), embed_vecs.cpu().detach().numpy(), axis=1)
        all_embed_vecs.append(embed_vecs)

    merged_embed_vecs = np.concatenate(all_embed_vecs, axis=0)
    np.savetxt(args.output_path, merged_embed_vecs, delimiter=",", fmt='%s')


if __name__ == '__main__':
    main()