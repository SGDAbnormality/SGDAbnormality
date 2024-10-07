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
import matplotlib as mpl

# Needed for pickle to work
from utils import InceptionNet, ConvModule, InceptionModule, DownsampleModule, CPU_Unpickler
from ratefunction import rate_function, eval_cummulant

plt.rcParams.update({
    'font.size': 10,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}\usepackage{bm}'
})
mpl.rc('font',family='Times New Roman')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.figsize'] = (16, 9)
fontsize = 40
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


labels = ["Standard", r'$\ell_2$ Regularized', "Initial"]

jet = plt.colormaps['Dark2']

models = []
with open("./models/inception_standard.pickle", "rb") as f:
  #models.append(pickle.load(f).to(device))
  models.append(CPU_Unpickler(f).load().to(device))

with open("./models/inception_l2.pickle", "rb") as f:
  #models.append(pickle.load(f).to(device))
  models.append(CPU_Unpickler(f).load().to(device))

models.append(InceptionNet(10, 3).to(device))


lambdas = np.arange(-0.2, 0.5, 0.01)
jensens = [eval_cummulant(model, lambdas, test_loader, device) for model in models]


plt.rcParams['figure.figsize'] = (14, 8)
for i in range(len(models)):
  plt.plot(lambdas, jensens[i], label=labels[i] ,linewidth=8, color = jet(i))
plt.grid()
plt.savefig("InceptionImgs/cummulant.pdf", format = "pdf",bbox_inches='tight')
plt.show()





s_values = np.arange(-0.5, 0.5, 0.01)

Is = [rate_function(model, s_values, device, test_loader)[:, 0] for model in models]


plt.rcParams['figure.figsize'] = (14, 8)
for i in range(len(Is)):
  plt.plot(s_values, Is[i], label=labels[i] ,linewidth=8, color = jet(i))
plt.ylim(-0.1,0.1)
plt.xlim(-0.5,0.5)
plt.grid()

plt.savefig("InceptionImgs/rates.pdf", format = "pdf",bbox_inches='tight')
plt.clf()

# Deviation plots

n = 50

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
    return np.array(losses)


train_losses = []
test_losses = []
train_losses_crop = []
test_losses_crop = []

for m in models:
  train_losses.append(eval(device, m, loader, criterion, 10))
  test_losses.append(np.mean(train_losses[-1]))


# Make histograms
plt.rcParams['figure.figsize'] = (14, 8)
plt.hist(train_losses[0], bins=20, alpha=0.7, label='Standard', color=jet(0))
# Vertical line at mean 
plt.axvline(x=test_losses[0], color=jet(0), linestyle='dashed', linewidth=8)
plt.hist(train_losses[1], bins=20, alpha=0.7, label=r'$\ell_2$ Regularized', color=jet(1))
plt.axvline(x=test_losses[1], color=jet(1), linestyle='dashed', linewidth=8)
plt.hist(train_losses[2], bins=20, alpha=0.7, label='Initial', color=jet(2))
plt.axvline(x=test_losses[2], color=jet(2), linestyle='dashed', linewidth=8)
plt.grid()
plt.savefig("InceptionImgs/losses.pdf", format = "pdf",bbox_inches='tight')

plt.clf()



width = 12
alphas = []
for i, m in enumerate(models):
  alphas.append(rate_function(m, test_losses[i] - train_losses[i], device, test_loader)[:, 0])
  
# Make histograms
plt.rcParams['figure.figsize'] = (14, 8)
alphas = np.array(alphas)
alphas_pos = [
  alphas[0][alphas[0] > 0],
  alphas[1][alphas[1] > 0],
  alphas[2][alphas[2] > 0]
]
alphas_neg = [
  alphas[0][alphas[0] < 0],
  alphas[1][alphas[1] < 0],
  alphas[2][alphas[2] < 0]
]
bins_pos = np.linspace(0.0025, 0.1, 20)
bins_neg = -bins_pos[::-1]
axis = plt.gca()
values_pos, bins_pos, _ = axis.hist(alphas_pos, bins=bins_pos, alpha=0.7, 
                           label=['Standard', r'Regularized $\ell_2$', 'Initial'], 
                           align = "left", color=[jet(0), jet(1), jet(2)], 
                           density=True)
values_neg, bins_neg, _ = axis.hist(alphas_neg, bins=bins_neg, alpha=0.7,
                            align = "right", color=[jet(0), jet(1), jet(2)], 
                            density=True)
min = np.min([np.min(bins_pos), np.min(bins_neg)])
max = np.max([np.max(bins_pos), np.max(bins_neg)])
x = np.linspace(-0.1, 0.1, 101)

axis.plot(x, n * np.exp(-n * np.abs(x)), linewidth=8, label=r"Exp(n)")
axis.set_xlim(-0.05, 0.05)
axis.grid()
plt.savefig("InceptionImgs/alphas.pdf", format = "pdf",bbox_inches='tight')
plt.show()

plt.cla()
lines, labels = axis.get_legend_handles_labels()
legend = plt.legend(lines, labels, loc=0, ncol=4, fancybox=True, shadow=True)
for line in legend.get_lines():
    line.set_linewidth(20)
def export_legend(legend, filename="InceptionImgs/legend.pdf"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

export_legend(legend)
plt.clf()