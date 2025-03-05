import json
import os
import sys
path = os.path.dirname(__file__)
path = os.path.dirname(path)
path = os.path.dirname(path)
sys.path.append(path)
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd

from tools.parser import creat_parser


class Sevir(data.Dataset):
    def __init__(self, configs, test=False, pre_train=False, vali=False):
        self.pre_train = pre_train
        self.test = test
        self.vali = vali
        self.mi = [0]
        self.ma = [80]
        self.configs = configs

        self.train_begin = 0
        self.vali_begin = 3000
        self.test_begin = 4000
        self.path = os.path.join(os.path.dirname(__file__), 'dataset')

    def __len__(self):
        if self.vali:
            return 1000
        elif self.test:
            return 1219
        else:
            return 3000

    def __getitem__(self, idx):

        if self.vali:
            idx = idx + self.vali_begin
        elif self.test:
            idx = idx + self.test_begin
        else:
            idx = idx

        path = os.path.join(self.path, f'sample_{idx}')
        path_vil = os.path.join(path, 'vil.npz')

        if self.pre_train:

            vil = np.load(path_vil)['data'].transpose([2, 0, 1])[
                    :self.configs.in_len + self.configs.pretrain_pred_len,
                    ...].reshape(
                self.configs.in_len + self.configs.pretrain_pred_len, 1,
                self.configs.input_size[0], self.configs.input_size[1])
            vil = torch.Tensor(vil)
            vil = self.norm(vil)
            return vil
        else:
            vil = np.load(path_vil)['data'].transpose([2, 0, 1])[
                    :self.configs.in_len + self.configs.pred_len,
                    ...].reshape(
                self.configs.in_len + self.configs.pred_len, 1,
                self.configs.input_size[0], self.configs.input_size[1])
            vil = torch.Tensor(vil)
            vil = self.norm(vil)
            return [vil[:self.configs.in_len, ...], vil[self.configs.in_len:self.configs.in_len+self.configs.pred_len, ...]]
        
    
    def norm(self, data):
        data = (data-0)/(80-0)
        return data
    
    def denorm(self, data):
        data = data*(80-0)+0
        return data



def main():
    parse = creat_parser()
    args = parse.parse_args()
    args.in_len = 10
    dataset = Sevir(args, pre_train=False, vali=False, test=False)
    for i in range(len(dataset)):
        vil = dataset[i][1][:, 0]
        if np.sum(vil.numpy()>0.4)/(vil.numpy().size) > 0.02:
            print(i)
        


if __name__ == '__main__':
    main()
