import os
import random

import torch
from torch.utils.data import Dataset

# 這兩個是資料處裡常用的套件
import numpy as np


class ntu_skeleton(Dataset):
    def __init__(self, data_path, mmap=True, window_size=99) -> None:
        super().__init__()
        self.data_path = data_path
        self.mmap = mmap
        self.window_size = window_size

        self.dataset = os.listdir(self.data_path)
    
    # def load_data(self):
    #     self.data_path
    #     skeletons = os.listdir(self.data_path)
    #     if self.mmap:
    #         self.data = [np.memmap(self.data_path + '/' + skeleton, dtype='float32',mode='r') for skeleton in skeletons]
    #     else:
    #         self.data = [np.load(self.data_path + '/' + skeleton) for skeleton in skeletons]

    def __getitem__(self, index):
        skeleton_name = self.dataset[index]
        data = np.load(self.data_path + '/' + skeleton_name, mmap_mode='r')
        # print(data.shape)
        data = self.auto_pading(data, self.window_size)
        data = torch.tensor(data)
        # print(data.size())
        data = data[None, :, :]
        # print(data.size())
        return data


    def auto_pading(self, data, size):
        # T: frame
        # V: vertix
        # print(data.shape)
        T, V = data.shape
        if T < size:
            # pad = T
            pad = 0
            # begin = random.randint(0, size - T) if random_pad else 0
            data_numpy_paded = np.zeros((size, V))
            # data_numpy_paded[begin:begin + T, :] = data
            # data_numpy_paded[:T, :] = data
            # while pad < size:
            #     data_numpy_paded[pad:, :] = data[, V]
            #     pad += T
            while pad + T <= size:
                data_numpy_paded[pad:pad+T, :] = data
                pad += T
            data_numpy_paded[pad:,:] = data[:size - pad, :]
            return data_numpy_paded
        else:
            return data[:size,:] if size > 0 else data


    def __len__(self):
        return len(self.dataset)
