# -*- coding: utf-8 -*-
# @Author  : HCN
# @Time    : 2024/12/21 15:57
# @File    : tid2013.py
# @Function:

import os

import torchvision
from PIL import Image
import torch
import numpy as np
import cv2
import pickle
import pandas as pd
from torch.utils.data import Dataset


class Csiq(Dataset):

    def __init__(self, transform=None, is_split=False, num_split=2, is_train=True):
        self.ref_path = r'E:\dataset\datasets\test\test\csiq'
        self.dist_path = r'E:\dataset\datasets\test\test\csiq'
        self.label_file = 'csiq_label.csv'
        # 数据集的取值范围
        self.max_score = 1
        self.min_score = 0

        self.transforms = transform
        # 读取label文件
        self.data_frame = pd.read_csv(self.label_file)

        # 如果分割，给出使用第几次分割的结果
        if is_split:
            self.num_split = num_split  # 使用第几次划分
            self.random_split_pkl = 'csiq_random20_num_split10.pkl'  # 10次随机划分文件
            with open(self.random_split_pkl, 'rb') as f:
                data = pickle.load(f)  # 字典
            self.random_split_xth = data[self.num_split]  # 第x次随机划分的字典, {'train':[], 'val':[]}

            if is_train:
                self.data_frame = self.data_frame.iloc[self.random_split_xth['train']]
            else:
                self.data_frame = self.data_frame.iloc[self.random_split_xth['val']]

        self.ref_name_list, self.dist_name_list, self.score_np = self.normalization(self.data_frame)

    def __getitem__(self, idx):
        ref_img_idx_path = os.path.join(self.ref_path, self.ref_name_list[idx])
        dist_img_idx_path = os.path.join(self.dist_path, self.dist_name_list[idx])
        label_idx = self.score_np[idx]

        ref_img_idx = Image.open(ref_img_idx_path).convert(mode='RGB')
        dist_img_idx = Image.open(dist_img_idx_path).convert(mode='RGB')

        if self.transforms:
            ref_img = self.transforms(ref_img_idx)
            dist_img = self.transforms(dist_img_idx)
            label = label_idx
        else:
            raise Exception("transform没有给定！！")

        return ref_img, dist_img, label

    def __len__(self):
        return len(self.ref_name_list)

    def normalization(self, data_frame):
        """
            DataFrame的列： ref_name, dist_name, mos(dmos)..
            将mos（dmos）归一化到0-1之间同时从dataframe中将三列单独拆开。
        Args:
            data_frame:

        Returns:
            list(ref_name) , list(dist_name)O, numpy(label)
        """
        ref_names_list = data_frame['ref_name'].to_list()
        dist_names_list = data_frame['dist_name'].to_list()
        score_np = data_frame.iloc[:, 2].to_numpy(dtype=np.float32)  # 读取第三列 ,转numpy
        score_np = (score_np - self.min_score) / (self.max_score - self.min_score)
        score_np = 1 - score_np
        return ref_names_list, dist_names_list, score_np


if __name__ == '__main__':
    from torchvision.transforms.functional import to_tensor

    transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(size=(256, 256)),
                                                 torchvision.transforms.ToTensor()
                                                 ])
    tid2013 = Csiq(transform=transforms, is_split=True, is_train=False) #174
    # tid2013 = Csiq(transform=transforms, is_split=True)  # 692
    # tid2013 = Csiq(transform=transforms)  # 886
    # ref_img, dist_img, label = tid2013.__getitem__(2)
    lens = tid2013.__len__()
    print(f"len is {lens}")