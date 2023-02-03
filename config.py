import os

import torch
import multiprocessing


class Config:
    en_channels = [1, 1, 128, 128, 64, 64, 32, 32, 16]
    de_channels = [16, 32, 32, 64, 64, 128, 128, 1, 1]
    en_stride = [1, 2, 1, 1, 2, 1, 2, 1]
    de_stride = [1, 2, 1, 2, 1, 1, 2, 1]
    optimizer = 'Adam'
    weight_decay = 0.0001
    base_lr = 0.1
    step = [10, 50]
    nesterov = True

    def __init__(self, args) -> None:
        self.dataset = args.dataset
        self.data_path = args.data_path
        self.batchnorm = args.batchnorm
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.gpu_ids = args.gpu_ids

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = min(multiprocessing.cpu_count() - 1, 20)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
