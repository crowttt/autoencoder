import os
import random

import torch
from torch.utils.data import Dataset

# 這兩個是資料處裡常用的套件
import numpy as np


class ntu_skeleton(Dataset):
    def __init__(self, data_path, mmap=True, window_size=128) -> None:
        super().__init__()
        self.data_path = data_path
        self.mmap = mmap
        self.window_size = window_size
        self.dataset = os.listdir(self.data_path)

    def __getitem__(self, index):
        skeleton_name = self.dataset[index]
        data = np.load(self.data_path + '/' + skeleton_name, mmap_mode='r')
        data = self.auto_pading(data, self.window_size)
        data = torch.tensor(data)
        data = data.flatten()
        return skeleton_name, data


    def auto_pading(self, data, size):
        # T: frame
        # V: vertix
        T, V = data.shape
        if T < size:
            pad = 0
            data_numpy_paded = np.zeros((size, V))
            while pad + T <= size:
                data_numpy_paded[pad:pad+T, :] = data
                pad += T
            data_numpy_paded[pad:,:] = data[:size - pad, :]
            return data_numpy_paded
        else:
            return data[:size,:] if size > 0 else data


    def __len__(self):
        return len(self.dataset)
