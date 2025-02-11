# -*- coding: utf-8 -*-
"""Abnormality SGD.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1q2XgFC4t7yBjUuVDyVx1DQULNdzU5BIW
"""


from __future__ import absolute_import, division, print_function, unicode_literals

import math

import numpy as np
import torch
from torch import optim

import torch.nn as nn
import pickle
from tqdm import tqdm

from train_test_loaders import getLoader_CIFAR_LENET
from train_eval import get_loss
from utils import list_files_in_directory
import argparse
import os
import sys

base_folder = "./"

"""Data"""


dataset_name = "CIFAR10"




parser = argparse.ArgumentParser()
parser.add_argument("-m", "--MODEL", help="MODEL", default= "MLP", type=str)
parser.add_argument("-b", "--BATCH_SIZE", help="BATCH_SIZE", default= 2500, type=int)
parser.add_argument("-i", "--N_ITERS", help="N_ITERS", default= 3000, type=int)
parser.add_argument("-s", "--N_STEPS", help="N_STEPS", default= 30, type=int)
parser.add_argument("-w", "--WEIGHT_DECAY", help="WEIGHT_DECAY", default= 0.0, type=float)

args = parser.parse_args()

network = args.MODEL
TRAIN_SIZE = 0
TEST_SIZE = 0
BATCH_SIZE = args.BATCH_SIZE
# Hyper-Parameters
RANDOM_SEED = 2147483647
LEARNING_RATE = 0.01
N_ITERS = args.N_ITERS
N_STEPS = args.N_STEPS
IMG_SIZE = 32
N_CLASSES = 10
WEIGHT_DECAY = args.WEIGHT_DECAY

name_model=f"{network}_{dataset_name}_{TRAIN_SIZE}_{TEST_SIZE}_{BATCH_SIZE}_{WEIGHT_DECAY}"

# setup devices
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.manual_seed(RANDOM_SEED)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    torch.cuda.manual_seed(RANDOM_SEED)
else:
    device = torch.device("cpu")

"""# Train"""

train_loader_batch, _ = getLoader_CIFAR_LENET(TRAIN_SIZE,TEST_SIZE, BATCH_SIZE, BATCH_SIZE, RANDOM_SEED)
train_loader_full, test_loader_full = getLoader_CIFAR_LENET(TRAIN_SIZE,TEST_SIZE, BATCH_SIZE, BATCH_SIZE, RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)


# Example usage
directory_path = f'./models/{name_model}/'
files = list_files_in_directory(directory_path)


g_cuda = torch.Generator(device='cpu')
g_cuda.manual_seed(RANDOM_SEED)


criterion = nn.CrossEntropyLoss().to(device)

criterion_nr = nn.CrossEntropyLoss(reduction="none").to(device)  # supervised classification loss

with open(directory_path + files[0], "rb") as handle:
    model = pickle.load(handle)

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

data_iter = iter(train_loader_batch)
iters_per_epoch = len(data_iter)
aux_loss = 1


# Create losses directory if it doesn't exist
os.makedirs(f'losses/{name_model}', exist_ok=True)


tq = tqdm(range(N_ITERS))



for it in tq:
    if (it % N_STEPS == 0 and it != 0):
        with open(directory_path + f"model_{int(it)}.pickle", "rb") as handle:
            model = pickle.load(handle)

        model.to(device)
        with torch.no_grad():
            # Forward pass
            test_loss = get_loss(device, model, test_loader_full).cpu().numpy()
            train_loss = get_loss(device, model, train_loader_full).cpu().numpy()
            
            # Save losses to CSV files using numpy
            np.savetxt(f'losses/{name_model}/test_losses_{int(it)}.csv', 
                      test_loss, 
                      delimiter=',', 
                      comments='')
            np.savetxt(f'losses/{name_model}/train_losses_{int(it)}.csv', 
                      train_loss, 
                      delimiter=',', 
                      comments='')




def load_losses(name_model, iteration):
    """
    Load train and test losses from CSV files for a specific model and iteration.
    
    Args:
        name_model (str): Name of the model directory
        iteration (int): Iteration number
        
    Returns:
        tuple: (train_losses, test_losses) as numpy arrays
    """
    test_losses = np.loadtxt(f'losses/{name_model}/test_losses_{iteration}.csv', delimiter=',')
    train_losses = np.loadtxt(f'losses/{name_model}/train_losses_{iteration}.csv', delimiter=',')
    return train_losses, test_losses

