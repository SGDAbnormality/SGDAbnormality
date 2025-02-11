import torch
from torch.utils.data import RandomSampler
from torchvision import transforms, datasets
import torchvision

import torch.nn as nn
import numpy as np



def getLoader_MNIST_MLP(TRAIN_SIZE=0, TEST_SIZE=0, BATCH_SIZE_TRAIN=100, BATCH_SIZE_TEST=1000, seed=0):

    class FlattenTransform:
        def __call__(self, tensor):
            return tensor.view(-1)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))#])
        ,FlattenTransform()])

    train_dataset = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform)

    test_dataset = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform)


    if TRAIN_SIZE == 0:
        TRAIN_SIZE = len(train_dataset)

    np.random.seed(seed)
    random_subset = np.random.choice(list(range(0, len(train_dataset))), TRAIN_SIZE, replace=False)
    train_dataset = torch.utils.data.Subset(train_dataset, random_subset)

    if TEST_SIZE == 0:
        TEST_SIZE = len(test_dataset)


    np.random.seed(seed)
    random_subset = np.random.choice(list(range(0, len(test_dataset))), TEST_SIZE, replace=False)
    test_dataset = torch.utils.data.Subset(test_dataset, random_subset)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                    batch_size=BATCH_SIZE_TEST,
                                                    shuffle=False)

    return train_loader, test_loader

def getLoader_MNIST_CONV(TRAIN_SIZE=0, TEST_SIZE=0, BATCH_SIZE_TRAIN=100, BATCH_SIZE_TEST=1000, seed=0):
    class FlattenTransform:
        def __call__(self, tensor):
            return tensor.view(-1)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform)

    test_dataset = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform)

    if TRAIN_SIZE == 0:
        TRAIN_SIZE = len(train_dataset)

    np.random.seed(seed)
    random_subset = np.random.choice(list(range(0, len(train_dataset))), TRAIN_SIZE, replace=False)
    train_dataset = torch.utils.data.Subset(train_dataset, random_subset)

    if TEST_SIZE == 0:
        TEST_SIZE = len(test_dataset)

    np.random.seed(seed)
    random_subset = np.random.choice(list(range(0, len(test_dataset))), TEST_SIZE, replace=False)
    test_dataset = torch.utils.data.Subset(test_dataset, random_subset)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                    batch_size=BATCH_SIZE_TEST,
                                                    shuffle=False)

    return train_loader, test_loader


def getLoader_FMNIST_CONV(TRAIN_SIZE=0, TEST_SIZE=0, BATCH_SIZE_TRAIN=100, BATCH_SIZE_TEST=1000, seed=0):
    class FlattenTransform:
        def __call__(self, tensor):
            return tensor.view(-1)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = torchvision.datasets.FashionMNIST(root='./data/', train=True, download=True, transform=transform)

    test_dataset = torchvision.datasets.FashionMNIST(root='./data/', train=False, download=True, transform=transform)

    if TRAIN_SIZE == 0:
        TRAIN_SIZE = len(train_dataset)

    np.random.seed(seed)
    random_subset = np.random.choice(list(range(0, len(train_dataset))), TRAIN_SIZE, replace=False)
    train_dataset = torch.utils.data.Subset(train_dataset, random_subset)

    if TEST_SIZE == 0:
        TEST_SIZE = len(test_dataset)

    np.random.seed(seed)
    random_subset = np.random.choice(list(range(0, len(test_dataset))), TEST_SIZE, replace=False)
    test_dataset = torch.utils.data.Subset(test_dataset, random_subset)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                    batch_size=BATCH_SIZE_TEST,
                                                    shuffle=False)

    return train_loader, test_loader


def getLoader_CIFAR_CONV(TRAIN_SIZE=0, TEST_SIZE=0, BATCH_SIZE_TRAIN=100, BATCH_SIZE_TEST=1000, seed=0):
    transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                                 torchvision.transforms.ToTensor()])

    train_dataset = datasets.CIFAR10(root='./data/',
                                    train=True,
                                    transform=transforms,
                                    download=True)


    test_dataset = datasets.CIFAR10(root='./data/',
                                    train=False,
                                    transform=transforms)

    if TRAIN_SIZE == 0:
        TRAIN_SIZE=len(train_dataset)

    np.random.seed(seed)
    random_subset = np.random.choice(list(range(0, len(train_dataset))), TRAIN_SIZE, replace=False)
    train_dataset = torch.utils.data.Subset(train_dataset, random_subset)

    if TEST_SIZE == 0:
        TEST_SIZE=len(test_dataset)

    np.random.seed(seed)
    random_subset = np.random.choice(list(range(0, len(test_dataset))), TEST_SIZE, replace=False)
    test_dataset = torch.utils.data.Subset(test_dataset, random_subset)

    g_cuda = torch.Generator(device='cpu')
    g_cuda.manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE_TRAIN, generator=g_cuda, shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                    batch_size=BATCH_SIZE_TEST,
                                                    shuffle=False)

    return train_loader, test_loader


def getLoader_CIFAR_LENET(TRAIN_SIZE=0, TEST_SIZE=0, BATCH_SIZE_TRAIN=100, BATCH_SIZE_TEST=1000, seed=0):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.CIFAR10(root='./data/',
                                    train=True,
                                    transform=transforms,
                                    download=True)


    test_dataset = datasets.CIFAR10(root='./data/',
                                    train=False,
                                    transform=transforms)

    if TRAIN_SIZE == 0:
        TRAIN_SIZE=len(train_dataset)

    if BATCH_SIZE_TRAIN == 0:
        BATCH_SIZE_TRAIN=len(train_dataset)

    np.random.seed(seed)
    random_subset = np.random.choice(list(range(0, len(train_dataset))), TRAIN_SIZE, replace=False)
    train_dataset = torch.utils.data.Subset(train_dataset, random_subset)

    if TEST_SIZE == 0:
        TEST_SIZE=len(test_dataset)

    if BATCH_SIZE_TEST == 0:
        BATCH_SIZE_TEST=len(test_dataset)


    np.random.seed(seed)
    random_subset = np.random.choice(list(range(0, len(test_dataset))), TEST_SIZE, replace=False)
    test_dataset = torch.utils.data.Subset(test_dataset, random_subset)

    g_cuda = torch.Generator(device='cpu')
    g_cuda.manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE_TRAIN, generator=g_cuda, shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                    batch_size=BATCH_SIZE_TEST,
                                                    shuffle=False)

    return train_loader, test_loader
