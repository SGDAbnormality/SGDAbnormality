import torch
from torch import nn


def createmodelLeNet5(k, random_seed, n_classes, n_channels):
    """ Create a LeNet5 model with k times the number of channels.

    Arguments
    ---------
    k : int
        Multiplies the number of channels in the layers of LeNet-5.
    random_seed : int
                  Random number for reproducibility.
    n_classes : int
                Number of classes in the dataset.
    n_channels : int
                 Number of channels in the input data.
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    return LeNet5(n_classes, n_channels, k)


class MLP(nn.Module):
    def __init__(self, input_size, output_size, L, W):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()

        self.flatten = nn.Flatten()

        # Input layer
        self.layers.append(nn.Linear(input_size, W))

        # Hidden layers
        for _ in range(L - 1):
            self.layers.append(nn.Linear(W, W))

        # Output layer
        self.layers.append(nn.Linear(W, output_size))

    def forward(self, x):
        x = self.flatten(x)
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)  # Output layer without activation
        return x


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
            nn.Linear(in_features=int(120 * k), out_features=int(84 * k)),
            nn.ReLU(),
            nn.Linear(in_features=int(84 * k), out_features=n_classes),
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