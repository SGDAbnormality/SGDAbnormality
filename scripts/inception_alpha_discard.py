import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import torchvision
from tqdm import tqdm

import pickle

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.figsize'] = (16, 9)
fontsize = 24
matplotlib.rcParams.update({'font.size': fontsize})
from matplotlib.pyplot import figure
import argparse
from train_test_loaders import getLoader_CIFAR_LENET
from models import createmodelLeNet5, InceptionNet, MLP
from utils import ensure_directory_exists


criterion = nn.CrossEntropyLoss() # supervised classification loss
criterion_nonreduced = nn.CrossEntropyLoss(reduce=False) # supervised classification loss

dataset_name = "CIFAR10"



parser = argparse.ArgumentParser()
parser.add_argument("-m", "--MODEL", help="MODEL", default= "MLP", type=str)
parser.add_argument("-b", "--BATCH_SIZE", help="BATCH_SIZE", default= 250, type=int)
parser.add_argument("-i", "--N_ITERS", help="N_ITERS", default= 3000, type=int)
parser.add_argument("-s", "--N_STEPS", help="N_STEPS", default= 150, type=int)
parser.add_argument("-w", "--WEIGHT_DECAY", help="WEIGHT_DECAY", default= 0.0, type=float)
parser.add_argument("-p", "--PROB", help="PROB", default= 0.01, type=float)
parser.add_argument("-e", "--N_EVAL", help="N_EVAL", default= 1, type=int)


args = parser.parse_args()

assert args.MODEL == "INCEPTION_DISCARD"

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

name_model=f"{network}_{dataset_name}_{TRAIN_SIZE}_{TEST_SIZE}_{BATCH_SIZE}_{args.PROB}"


# setup devices
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.manual_seed(RANDOM_SEED)
else:
    device = torch.device("cpu")


#######################################################################
############################# DATASET #################################
#######################################################################

train_loader_batch, test_loader_batch = getLoader_CIFAR_LENET(TRAIN_SIZE, TEST_SIZE, BATCH_SIZE, 500, RANDOM_SEED)

model = InceptionNet(10,3).to(device)

def get_log_p(device, model, loader):
    cce = nn.CrossEntropyLoss(reduction = "none") # supervised classification loss
    model.eval()
    aux = []
    with torch.no_grad():
      for data, targets in loader:
          data = data.to(device)
          targets = targets.to(device)
          logits = model(data)
          log_p = -cce(logits, targets) # supervised loss
          aux.append(log_p)
    return torch.cat(aux)

#Binary Search for lambdas
def rate_function_BS(log_p, s_value):
  if (s_value<0):
    min_lamb=torch.tensor(-10000).to(device)
    max_lamb=torch.tensor(0).to(device)
  else:
    min_lamb=torch.tensor(0).to(device)
    max_lamb=torch.tensor(10000).to(device)

  s_value=torch.tensor(s_value).to(device)
  return aux_rate_function_TernarySearch(log_p, s_value, min_lamb, max_lamb, 0.01)

def eval_log_p(log_p, lamb, s_value):
  jensen_val=(torch.logsumexp(lamb * log_p, 0) - torch.log(torch.tensor(log_p.shape[0], device = device)) - lamb *torch.mean(log_p))
  return lamb*s_value - jensen_val

def aux_rate_function_BinarySearch(log_p, s_value, low, high, epsilon):

  while (high - low) > epsilon:
      mid = (low + high) / 2
      print(mid)
      print(eval_log_p(log_p, low, s_value))
      print(eval_log_p(log_p, mid, s_value))
      print(eval_log_p(log_p, high, s_value))
      print("--")
      if eval_log_p(log_p, mid, s_value) < eval_log_p(log_p, high, s_value):
          low = mid
      else:
          high = mid

  # Return the midpoint of the final range
  mid = (low + high) / 2
  return [eval_log_p(log_p, mid, s_value).detach().cpu().numpy(), mid.detach().cpu().numpy(), (mid*s_value - eval_log_p(log_p, mid, s_value)).detach().cpu().numpy()]


def aux_rate_function_TernarySearch(log_p, s_value, low, high, epsilon):

  while (high - low) > epsilon:
    mid1 = low + (high - low) / 3
    mid2 = high - (high - low) / 3

    if eval_log_p(log_p, mid1, s_value) < eval_log_p(log_p, mid2, s_value):
        low = mid1
    else:
        high = mid2

  # Return the midpoint of the final range
  mid = (low + high) / 2
  return [eval_log_p(log_p, mid, s_value).detach().cpu().numpy(), mid.detach().cpu().numpy(), (mid*s_value - eval_log_p(log_p, mid, s_value)).detach().cpu().numpy()]

import math
def aux_rate_function_golden_section_search(log_p, s_value, a, b, epsilon):
    """
    Maximizes a univariate function using the golden section search algorithm.

    Parameters:
        f (function): The function to minimize.
        a (float): The left endpoint of the initial search interval.
        b (float): The right endpoint of the initial search interval.
        tol (float): The error tolerance value.

    Returns:
        float: The x-value that minimizes the function f.
    """
    # Define the golden ratio
    golden_ratio = (torch.sqrt(torch.tensor(5).to(device)) - 1) / 2

    # Define the initial points
    c = b - golden_ratio * (b - a)
    d = a + golden_ratio * (b - a)

    # Loop until the interval is small enough
    while abs(c - d) > epsilon:
        # Compute the function values at the new points
        fc = eval_log_p(log_p, c, s_value)
        fd = eval_log_p(log_p, d, s_value)

        # Update the interval based on the function values
        if fc > fd:
            b = d
            d = c
            c = b - golden_ratio * (b - a)
        else:
            a = c
            c = d
            d = a + golden_ratio * (b - a)

    # Return the midpoint of the final interval
    mid = (a + b) / 2
    return [eval_log_p(log_p, mid, s_value).detach().cpu().numpy(), mid.detach().cpu().numpy(), (mid*s_value - eval_log_p(log_p, mid, s_value)).detach().cpu().numpy()]

def eval_jensen(model, lambdas):
  log_p = get_log_p(device, model, test_loader_batch)
  return np.array(
      [
          (torch.logsumexp(lamb * log_p, 0) - torch.log(torch.tensor(log_p.shape[0], device = device)) - torch.mean(lamb * log_p)).detach().cpu().numpy() for lamb in lambdas
       ])

def rate_function(model, lambdas, s_values):
  jensen_vals = eval_jensen(model, lambdas)
  return np.array([ np.max(lambdas*s - jensen_vals) for s in s_values]), np.array([lambdas[np.argmax(lambdas*s - jensen_vals)] for s in s_values])

def inverse_rate_function(model, lambdas, rate_vals):
  jensen_vals = eval_jensen(model, lambdas)

  return np.array([ np.min((jensen_vals + rate)/lambdas) for rate in rate_vals])

def eval(device, model, loader, criterion):
    correct = 0
    total = 0
    losses = []
    model.eval()
    with torch.no_grad():
        for data, targets in loader:
            total += targets.size(0)
            data = data.to(device)
            targets = targets.to(device)
            logits = model(data)
            probs = F.softmax(logits, dim=1)
            predicted = torch.argmax(probs, 1)
            correct += (predicted == targets).sum().detach().cpu().numpy()

            loss = criterion(logits, targets) # supervised loss
            losses.append(loss.detach().cpu().numpy())

    return correct, total, np.mean(losses), np.var(losses)


directory_path = f"models/{name_model}/"
ensure_directory_exists(directory_path)


def train(model, train_loader, SKIP=2):

    test_loss = []
    train_loss = []
    batch_loss = []
    discarded = []

    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    #optimizer = optim.Adam(model.parameters(), lr=1e-3)
    data_iter = iter(train_loader)


    it = 0

    with tqdm(total=N_ITERS) as pbar:
        while True:
            model.train()

            try:
                inputs, target = next(data_iter)
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                data_iter = iter(train_loader)
                inputs, target = next(data_iter)

            inputs = inputs.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            logits = model(inputs) # forward pass

            loss = criterion(logits, target) # supervised loss
            loss_aux = loss.detach().cpu().numpy()

            if it % args.N_EVAL == 0 and it >= 30:

                model.eval()

                log_p = get_log_p(device, model, test_loader_batch)

                test_loss.append(-np.mean(log_p.detach().cpu().numpy()))

                pbar.set_postfix(
                    {"Batch loss": loss_aux, "Test loss": test_loss[-1]}
                )


                with open(f'{directory_path}/model_{it}.pickle', 'wb') as handle:
                    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

            model.train()

            if it >= 30:

              if test_loss[-1] < loss_aux:
                alpha = 0
              else:
                alpha = rate_function_BS(log_p, test_loss[-1]-loss_aux)[0]
              print(test_loss[-1], loss_aux, alpha)
              print(alpha, np.log(1.0/args.PROB) / BATCH_SIZE)

              if SKIP==2 and alpha < np.log(1.0/args.PROB) / BATCH_SIZE:
                loss.backward() # computes gradients
                optimizer.step()
              else:
                discarded.append(it)

            else:
              loss.backward() # computes gradients
              optimizer.step()

            pbar.update(1)
            it = it + 1
            if it == N_ITERS:
                break

    print("Discarded: ", discarded)
    np.savetxt(f"{directory_path}/discarded_iterations.txt", np.array(discarded))


train(model, train_loader_batch, SKIP=2)