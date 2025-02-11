
import torch.nn as nn
import torch
from torchvision import datasets, transforms
import numpy as np
import torchvision
import pickle

import ssl

from models import createmodelLeNet5, InceptionNet, MLP
from train_eval import train_SuperBatch
from train_test_loaders import getLoader_CIFAR_LENET

ssl._create_default_https_context = ssl._create_unverified_context

from utils import ensure_directory_exists

import argparse


# Create custom loss functions
criterion = nn.CrossEntropyLoss() # supervised classification loss


dataset_name = "CIFAR10"



parser = argparse.ArgumentParser()
parser.add_argument("-m", "--MODEL", help="MODEL", default= "INCEPTION", type=str)
parser.add_argument("-b", "--BATCH_SIZE", help="BATCH_SIZE", default= 500, type=int)
parser.add_argument("-i", "--N_ITERS", help="N_ITERS", default= 3000, type=int)
parser.add_argument("-s", "--N_STEPS", help="N_STEPS", default= 150, type=int)
parser.add_argument("-w", "--WEIGHT_DECAY", help="WEIGHT_DECAY", default= 0.1, type=float)


args = parser.parse_args()

network = args.MODEL
n_models = 10
TRAIN_SIZE = 0
TEST_SIZE = 0
BATCH_SIZE = args.BATCH_SIZE
# Hyper-Parameters
RANDOM_SEED = 2147483647
LEARNING_RATE = 0.001
N_ITERS = args.N_ITERS
N_STEPS = args.N_STEPS
IMG_SIZE = 32
N_CLASSES = 10
WEIGHT_DECAY = args.WEIGHT_DECAY

name_model=f"{network}_{dataset_name}_{TRAIN_SIZE}_{TEST_SIZE}_{BATCH_SIZE}_{WEIGHT_DECAY}"

#LEARNING_RATE = LEARNING_RATE*BATCH_SIZE/200
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

#######################################################################
############################# DATASET #################################
#######################################################################

train_loader_batch, _ = getLoader_CIFAR_LENET(TRAIN_SIZE,TEST_SIZE, 50, 50, RANDOM_SEED)

#######################################################################
############################# TRAIN MODELS ############################
#######################################################################

# Initialize channels for models
ks = [3]

# Initialice models
#models = [createmodelLeNet5(k, RANDOM_SEED, 10, 3).to(device) for k in ks]
if args.MODEL == "INCEPTION":
    models = [InceptionNet(10,3).to(device)]
elif args.MODEL == "MLP":
    models = [MLP(32*32*3,10, 2, 512).to(device)]

n_params = []
for model in models:
  n = 0
  for parameter in model.parameters():
    n += parameter.flatten().size(0)
  n_params.append(n)

if args.MODEL == "INCEPTION":
    labels = ["ConvNN-"+str(p//1000)+"k" for p in n_params]
elif args.MODEL == "MLP":
    labels = ["MLP-"+str(p//1000)+"k" for p in n_params]

np.savetxt("models/model_labels.txt",labels, delimiter=" ", fmt="%s")
np.savetxt("models/n_params.txt",n_params)



for i in range(len(models)):
  g_cuda = torch.Generator(device='cpu')
  g_cuda.manual_seed(RANDOM_SEED)

  directory_path = f"models/{name_model}/"
  ensure_directory_exists(directory_path)

  train_SuperBatch(directory_path, models[i], train_loader_batch, BATCH_SIZE, LEARNING_RATE, N_ITERS, WEIGHT_DECAY, device)
  
