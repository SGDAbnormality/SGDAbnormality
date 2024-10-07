import numpy as np
import torch
from torch import optim as optim, nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import pickle

from utils import EarlyStopper


def train_SuperBatch(folder_name, model, train_loader, super_batch_size, learning_rate, n_iters, weight_decay, device):
    """ Train the model using SGD with ExponentialLR scheduler and EarlyStopper

    Arguments
    ---------
    model : torch.nn.Module
        The model to train
    train_loader : torch.utils.data.DataLoader
        The data loader for the training set
    learning_rate : float
        The learning rate for the optimizer
    n_iters : int
        The number of iterations to train for
    device : torch.device
        The device to train on
    criterion : torch.nn.Module
        The loss function to optimize
    """

    cce = nn.CrossEntropyLoss(reduction="sum")  # supervised classification loss

    # Initialize EarlyStopper, optimizer and scheduler
    es = EarlyStopper(patience=2)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # Initialize data iterator
    data_iter = iter(train_loader)
    iters_per_epoch = len(data_iter)
    aux_loss = 1
    batch_loss = 0
    # Zero the gradients
    optimizer.zero_grad()
    n_samples = 0

    it = 0

    with tqdm(total=n_iters) as pbar:
        while True:
            # Set model to train mode
            model.train()

            # Get inputs and targets. If loader is exhausted, reinitialize.
            try:
                inputs, target = next(data_iter)
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                data_iter = iter(train_loader)
                inputs, target = next(data_iter)

            # Move data to device
            inputs = inputs.to(device)
            target = target.to(device)

            # Forward pass
            logits = model(inputs)

            # Compute the loss
            loss = cce(logits, target) / super_batch_size
            aux_loss += loss.detach().cpu().numpy()
            batch_loss += loss.detach().cpu().numpy()

            # Backward pass
            loss.backward()

            n_samples += train_loader.batch_size
            if n_samples >= super_batch_size:
                if weight_decay > 0:
                    # Add L2 regularization to the loss
                    l2_reg = torch.tensor(0.).to(device)
                    for param in model.parameters():
                        l2_reg += torch.norm(param)
                    loss = weight_decay * l2_reg
                    # Backward pass
                    loss.backward()

                # Log the loss
                pbar.set_postfix(
                    {"Train cce": batch_loss, "Patience": es.counter}
                )
                batch_loss = 0

                pbar.update(1)

                n_samples = 0

                # Update the weights
                optimizer.step()

                # Zero the gradients
                optimizer.zero_grad()

                it = it + 1
                if it == n_iters:
                    break

            # Step the scheduler and check for early stopping
            if it % 10 == 0 and it != 0:
                with open(f'{folder_name}/model_{it}.pickle', 'wb') as handle:
                    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Step the scheduler and check for early stopping
            #if it % iters_per_epoch == 0 and it != 0:
            #    scheduler.step()
                #if aux_loss / iters_per_epoch < 0.015:# or es.early_stop(aux_loss):
                #    break
                #aux_loss = 0

    with open(f'{folder_name}/model_final.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return model

def train(model, train_loader, learning_rate, n_iters, weight_decay, device, criterion):
    """ Train the model using SGD with ExponentialLR scheduler and EarlyStopper

    Arguments
    ---------
    model : torch.nn.Module
        The model to train
    train_loader : torch.utils.data.DataLoader
        The data loader for the training set
    learning_rate : float
        The learning rate for the optimizer
    n_iters : int
        The number of iterations to train for
    device : torch.device
        The device to train on
    criterion : torch.nn.Module
        The loss function to optimize
    """
    # Initialize EarlyStopper, optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize data iterator
    data_iter = iter(train_loader)
    iters_per_epoch = len(data_iter)
    aux_loss = 1

    tq = tqdm(range(n_iters))
    for it in tq:
        # Set model to train mode
        model.train()

        # Get inputs and targets. If loader is exhausted, reinitialize.
        try:
            inputs, target = next(data_iter)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            data_iter = iter(train_loader)
            inputs, target = next(data_iter)

        # Move data to device
        inputs = inputs.to(device)
        target = target.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(inputs)

        # Compute the loss
        loss = criterion(logits, target)
        aux_loss += loss.detach().cpu().numpy()

        # Log the loss
        tq.set_postfix(
            {"Train cce": loss.detach().cpu().numpy()}
        )

        if weight_decay>0:
            # Add L2 regularization to the loss
            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += weight_decay * l2_reg

        # Backward pass
        loss.backward()
        # Update the weights
        optimizer.step()

        # Step the scheduler and check for early stopping
        if it % iters_per_epoch == 0 and it != 0:
            if aux_loss / iters_per_epoch < 0.01:
                break
            aux_loss = 0

    return model


def eval_detach(device, model, loader, criterion):
    """ Evaluate the model on the loader using the criterion.

    Arguments
    ---------
    device : torch.device
        The device to evaluate on
    model : torch.nn.Module
        The model to evaluate
    loader : torch.utils.data.DataLoader
        The data loader to evaluate on
    criterion : torch.nn.Module
        The loss function to evaluate with
    """

    # Initialize counters
    correct = 0
    total = 0
    losses = 0


    for data, targets in loader:

        # Move data to device
        total += targets.size(0)
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass
        logits = model(data)

        # Compute the loss
        loss = criterion(logits, targets)
        # Update the loss
        losses += loss.detach().cpu().numpy() * targets.size(0)

    return losses / total


def eval(device, model, loader, criterion):
    """ Evaluate the model on the loader using the criterion.

    Arguments
    ---------
    device : torch.device
        The device to evaluate on
    model : torch.nn.Module
        The model to evaluate
    loader : torch.utils.data.DataLoader
        The data loader to evaluate on
    criterion : torch.nn.Module
        The loss function to evaluate with
    """

    # Initialize counters
    correct = 0
    total = 0
    losses = 0

    # Set model to evaluation mode
    model.eval()
    model = model.to(device)

    # Iterate over the loader
    with torch.no_grad():
        for data, targets in loader:

            # Move data to device
            total += targets.size(0)
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(data)
            # Compute the probabilities
            probs = F.softmax(logits, dim=1)
            # Get the predicted class
            predicted = torch.argmax(probs, 1)
            # Update the counters
            correct += (predicted == targets).sum().detach().cpu().numpy()

            # Compute the loss
            loss = criterion(logits, targets)
            # Update the loss
            losses += loss.detach().cpu().numpy() * targets.size(0)

    return correct, total, losses / total


def get_loss_samples(device, model, loader, samples, expected = None):
    cce = nn.CrossEntropyLoss(reduction="mean")  # supervised classification loss
    model.eval()
    aux = []
    data_iter = iter(loader)
    with tqdm(total=samples) as pbar:
        with torch.no_grad():
            while len(aux) < samples:
                try:
                    data, targets = next(data_iter)
                except StopIteration:
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader
                    data_iter = iter(loader)
                    data, targets = next(data_iter)

                data = data.to(device)
                targets = targets.to(device)
                logits = model(data)
                loss = cce(logits, targets)  # supervised loss
                aux.append(loss.detach().cpu().numpy())
                pbar.update(1)
    return np.stack(aux)


def get_loss(device, model, loader):
    cce = nn.CrossEntropyLoss(reduction="none")  # supervised classification loss
    model.eval()
    aux = []
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)
            logits = model(data)
            loss = cce(logits, targets)  # supervised loss
            aux.append(loss)
    return torch.cat(aux)


def get_data_set_grad(device, model, loader):

    model.eval()

    total_size = len(loader.dataset)

    cce = nn.CrossEntropyLoss(reduction="sum")  # supervised classification loss
    aux = []
    aux_loss = 0
    for data, targets in loader:
        data = data.to(device)
        targets = targets.to(device)
        logits = model(data)
        loss = cce(logits, targets)/total_size
        loss.backward()
        aux_loss += loss.detach().cpu().numpy()
    return aux_loss

def get_loss_grad(device, model, loader):
    model.eval()
    cce = nn.CrossEntropyLoss(reduction="none")  # supervised classification loss
    aux = []
    for data, targets in loader:
        data = data.to(device)
        targets = targets.to(device)
        logits = model(data)
        loss = cce(logits, targets)
        aux.append(loss)

    return torch.cat(aux)

def jensen_val_grad(device, model, loader, sign, lamb, all_losses):

    constant = torch.logsumexp(-sign * lamb * all_losses, 0)

    cce = nn.CrossEntropyLoss(reduction="none")  # supervised classification loss

    for data, targets in loader:
        data = data.to(device)
        targets = targets.to(device)
        logits = model(data)
        losses = cce(logits, targets)
        factor = torch.exp(torch.logsumexp(-sign * lamb * losses, 0)-constant)/lamb
        factor = factor.detach()
        jensen = torch.logsumexp(-sign.detach() * lamb * losses, 0)*factor
        jensen.backward()
