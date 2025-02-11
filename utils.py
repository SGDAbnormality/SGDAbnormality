import io
import pickle

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm


import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl


import os


class LeNet5(nn.Module):
    def __init__(self, n_classes, input_channels, k):
        """ Initialize the LeNet-5 model with k times the number of channels.
        
        The model has 3 convolutional layers, 2 pooling layers, and 2 fully connected layers.
        The first convolutional layer has int(6k) output channels.
        The second convolutional layer has int(16k) output channels.
        The third convolutional layer has int(120k) output channels.
        
        Arguments
        ---------
        n_classes : int
            Number of classes in the dataset.
        input_channels : int
            Number of channels in the input data.
        k : int
            Multiplicative factor of the number of channels.
        
        """
        super(LeNet5, self).__init__()

        self.part1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=int(6 * k),
                kernel_size=5,
                stride=1,
            ),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )
        self.part2 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(6 * k),
                out_channels=int(16 * k),
                kernel_size=5,
                stride=1,
            ),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )
        self.part3 = nn.Sequential(
            nn.Conv2d(
                in_channels=int(16 * k),
                out_channels=int(120 * k),
                kernel_size=5,
                stride=1,
            ),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=int(120 * k), out_features=int(84)),
            nn.ReLU(),
            nn.Linear(in_features=int(84), out_features=n_classes),
        )

    def forward(self, x):
        x = self.part1(x)
        x = self.part2(x)
        x = self.part3(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvModule, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class InceptionModule(nn.Module):

    def __init__(self, in_channels, f_1x1, f_3x3):
        super(InceptionModule, self).__init__()

        self.branch1 = nn.Sequential(
            ConvModule(in_channels, f_1x1, kernel_size=1, stride=1, padding=0)
        )

        self.branch2 = nn.Sequential(
            ConvModule(in_channels, f_3x3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        return torch.cat([branch1, branch2], 1)


class DownsampleModule(nn.Module):
    def __init__(self, in_channels, f_3x3):
        super(DownsampleModule, self).__init__()

        self.branch1 = nn.Sequential(ConvModule(in_channels, f_3x3, kernel_size=3, stride=2, padding=0))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        return torch.cat([branch1, branch2], 1)

class InceptionNet(nn.Module):
    def __init__(self, num_classes, input_channels):
        super().__init__()

        self.conv1 = ConvModule(in_channels =input_channels,out_channels=96, kernel_size=3, stride=1, padding=0)
        self.inception1 = InceptionModule(in_channels=96,f_1x1=32,f_3x3=32)
        self.inception2 = InceptionModule(in_channels=64,f_1x1=32,f_3x3=48)
        self.down1 = DownsampleModule(in_channels=80,f_3x3=80)
        self.inception3 = InceptionModule(in_channels=160,f_1x1=112,f_3x3=48)
        self.inception4 = InceptionModule(in_channels=160,f_1x1=96,f_3x3=64)
        self.inception5 = InceptionModule(in_channels=160,f_1x1=80,f_3x3=80)
        self.inception6 = InceptionModule(in_channels=160,f_1x1=48,f_3x3=96)
        self.down2 = DownsampleModule(in_channels=144,f_3x3=96)
        self.inception7 = InceptionModule(in_channels=240,f_1x1=176,f_3x3=160)
        self.inception8 = InceptionModule(in_channels=336,f_1x1=176,f_3x3=160)
        self.meanpool = nn.AdaptiveAvgPool2d((7,7))
        self.fc = nn.Linear(16464, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.down1(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.inception6(x)
        x = self.down2(x)
        x = self.inception7(x)
        x = self.inception8(x)
        x = self.meanpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x


def find_change_point(data):
    # Smooth the data using a simple moving average to reduce noise
    window_size = 5  # You can adjust this depending on the noise level and data size
    smoothed_data = np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    # Find the index of the minimum value in the smoothed data
    change_point = np.argmin(smoothed_data) + window_size // 2  # adjust index due to smoothing

    return change_point, smoothed_data

def list_files_in_directory(directory):
    # Get a list of all files and directories in the specified directory
    all_items = os.listdir(directory)

    # Filter out directories, keeping only files
    files = [item for item in all_items if os.path.isfile(os.path.join(directory, item))]

    return files


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")

# Example usage
def non_latex_format():
    """ Set the matplotlib style to non-latex format """
    mpl.rcParams.update(mpl.rcParamsDefault)

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    plt.rcParams["figure.figsize"] = (16, 9)
    fontsize = 26
    matplotlib.rcParams.update({"font.size": fontsize})


def latex_format():
    """ Set the matplotlib style to latex format """
    plt.rcParams.update(
        {
            "font.size": 10,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsfonts}\usepackage{bm}",
        }
    )
    mpl.rc("font", family="Times New Roman")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    plt.rcParams["figure.figsize"] = (16, 9)
    fontsize = 30
    matplotlib.rcParams.update({"font.size": fontsize})




class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        """ Initialize the EarlyStopper object.
        
        Arguments
        ---------
        patience : int
            The number of iterations to wait before stopping training.
        min_delta : float
            The minimum delta between the current loss and the best loss.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        """ Check if the training should stop.
        
        Arguments
        ---------
        validation_loss : float
            The loss on the validation set.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

