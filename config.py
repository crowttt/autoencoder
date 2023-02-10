import os

import torch
import multiprocessing


class Config:
    kernel = 3
    en_channels = [9600, 2400, 1200, 300, 75]
    de_channels = [75, 300, 1200, 2400, 9600]
    en_stride = [1, 2, 1, 1, 2, 1, 2, 1]
    de_stride = [1, 2, 1, 2, 1, 1, 2, 1]
    optimizer = 'Adam'
    weight_decay = 0.00001
    base_lr = 0.000001
    step = [10, 50]
    nesterov = True

    def __init__(self, args=None) -> None:
        if args:
            self.dataset = args.dataset
            self.data_path = args.data_path
            self.epochs = args.epochs
            self.batch_size = args.batch_size
            self.gpu_ids = args.gpu_ids
            self.pre_encoder = args.pretrained_encoder
            self.pre_decoder = args.pretrained_decoder

            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = min(multiprocessing.cpu_count() - 1, 20)
