import math

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from scipy import stats
from torchvision import datasets, transforms
import numpy as np
import torchvision
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
import sys

sys.path.append(".")

# Needed for pickle to work
from utils import ConvModule, InceptionModule, DownsampleModule, CPU_Unpickler
from ratefunction import rate_function, eval_cummulant
from models import InceptionNet

plt.rcParams.update({
    'font.size': 10,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}\usepackage{bm}'
})
mpl.rc('font',family='Times New Roman')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.figsize'] = (16, 9)
fontsize = 30
matplotlib.rcParams.update({'font.size': fontsize})


criterion = nn.CrossEntropyLoss() # supervised classification loss
criterion_nonreduced = nn.CrossEntropyLoss(reduce=False) # supervised classification loss


# LENET parameters
RANDOM_SEED = 2147483647
LEARNING_RATE = 0.01 #0.0001 for MLP
SUBSET_SIZE = 50000
TEST_SUBSET_SIZE = 10000
N_ITERS = 30000
BATCH_SIZE = 200
BATCH_SIZE_TEST = 1000
IMG_SIZE = 32
N_CLASSES = 10



# setup devices
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.manual_seed(RANDOM_SEED)
else:
    device = torch.device("cpu")


transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                 torchvision.transforms.ToTensor()])

train_dataset = datasets.CIFAR10(root='cifar_data',
                                train=True,
                                transform=transforms,
                                download=True)



test_dataset = datasets.CIFAR10(root='cifar_data',
                                train=False,
                                transform=transforms)

test_dataset = torch.utils.data.Subset(test_dataset, list(range(0, TEST_SUBSET_SIZE)))
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                          batch_size=BATCH_SIZE_TEST,
                          shuffle=False)


labels = ["Standard", "L2-Crop", "Initial"]

jet = plt.colormaps['Dark2']

models = []
with open("./models/inception_standard.pickle", "rb") as f:
  #models.append(pickle.load(f).to(device))
  models.append(CPU_Unpickler(f).load().to(device))

with open("./models/inception_l2_crop.pickle", "rb") as f:
  #models.append(pickle.load(f).to(device))
  models.append(CPU_Unpickler(f).load().to(device))

models.append(InceptionNet(10, 3).to(device))



# Deviation plots

n = 2000
epochs = 1

g_cuda = torch.Generator(device='cpu')
g_cuda.manual_seed(RANDOM_SEED)
loader = torch.utils.data.DataLoader(dataset=test_dataset,
                          batch_size=n,
                          generator=g_cuda,
                          shuffle=True)

@torch.no_grad()
def eval(device, model, loader, criterion, epochs = 1):
    losses = []
    model.eval()
    
    with tqdm(total=epochs * len(loader), desc="Evaluating", position=0, leave=True) as pbar:
        for _ in range(epochs):
          for data, targets in loader:
              data = data.to(device)
              targets = targets.to(device)
              logits = model(data)

              loss = criterion(logits, targets) # supervised loss
              losses.append(loss.detach().cpu().numpy())
              pbar.update(1)
    return np.concatenate(losses)


train_losses = []
test_losses = []
train_losses_crop = []
test_losses_crop = []

for m in models:
  train_losses.append(eval(device, m, loader, criterion_nonreduced, epochs))
  train_losses[-1] = train_losses[-1] - np.mean(train_losses[-1])

num_samples = 250
vals = []
for m in models:
    val_model=[]
    for _ in range(10000):
        val_model.append(np.mean(np.random.choice(train_losses[0], size=num_samples, replace=False)))
    val_model = np.array(val_model)
    val_model = val_model/ np.std(val_model)
    vals.append(val_model)

for m in range(len(models)):
    for k in range(2,5):
        print(f"Model {m} - {k}-Cumlant/{k}!: {stats.kstat(vals[m], k)/math.factorial(k)}")
    print()
# Make histograms
plt.rcParams['figure.figsize'] = (14, 8)
plt.hist(train_losses[0], bins=20, alpha=0.7, label='Standard', color=jet(0))
# Vertical line at mean 
plt.axvline(x=test_losses[0], color=jet(0), linestyle='dashed', linewidth=2)
plt.hist(train_losses[1], bins=20, alpha=0.7, label='L2-Crop', color=jet(1))
plt.axvline(x=test_losses[1], color=jet(1), linestyle='dashed', linewidth=2)
plt.hist(train_losses[2], bins=20, alpha=0.7, label='Initial', color=jet(2))
plt.axvline(x=test_losses[2], color=jet(2), linestyle='dashed', linewidth=2)
plt.legend()
plt.grid()
plt.savefig("InceptionImgs/losses.pdf", format = "pdf",bbox_inches='tight')

plt.clf()



