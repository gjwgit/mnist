from argparse import ArgumentParser
import logging 
import numpy as np 
import os 

import torch 
from torchvision import transforms 
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader 

def dataloader(batch_size):
    '''downloads MNIST dataset, transforms data and returns train and val loaders for running model''' 
    data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    tr_set = MNIST(download=True, root=".", transform=data_transform, train=True)
    val_set = MNIST(download=False, root=".", transform=data_transform, train=False)
    train_loader = DataLoader(tr_set,batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set,batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
