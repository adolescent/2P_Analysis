# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 13:52:35 2022

@author: ZR
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
#%% Load data sets.
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64# how many batches you want split all data into.

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")# X is each single batch of data.[Sample_Num,?,height,width]
    print(f"Shape of y: {y.shape} {y.dtype}")# y is the label index of each graph in the batch.
    break # you only need 1 batch.



