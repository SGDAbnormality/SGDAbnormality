

from __future__ import absolute_import, division, print_function, unicode_literals
import datetime
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import numpy as np
import pickle
import matplotlib
import sys
sys.path.append('.')
from ratefunction import rate_function, LeNet5
from train_eval import train, eval, get_loss_samples

base_folder = "./Fig2"

""" Set the matplotlib style to latex format """
plt.rcParams.update(
    {
        "font.size": 10,
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsfonts}\usepackage{bm}",
    }
)
matplotlib.rc("font", family="Times New Roman")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["figure.figsize"] = (16, 9)
fontsize = 30
matplotlib.rcParams.update({"font.size": fontsize})

batch_losses_50_0 = np.loadtxt("Fig2\samples\BatchLosses50-Models0.txt")
batch_losses_50_1 = np.loadtxt("Fig2\samples\BatchLosses50-Models1.txt")
batch_losses_500_0 = np.loadtxt("Fig2\samples\BatchLosses500-Models0.txt")
batch_losses_500_1 = np.loadtxt("Fig2\samples\BatchLosses500-Models1.txt")

Ltest0 = np.loadtxt("Fig2\Ltest\L_test_SGD-50-0.txt")
Ltest1 = np.loadtxt("Fig2\Ltest\L_test_SGD-50-1.txt")


jet = matplotlib.colormaps["Set2"]
colors = [jet(0), jet(1), jet(2), jet(3)]
plt.hist([batch_losses_50_0, batch_losses_50_1, batch_losses_500_0, batch_losses_500_1],
         density=True,
         align="mid",
         bins=40,
         color = colors,
         label = [r"$\hat{L}(D_{50},\bm{\theta}_1)$",
                  r"$\hat{L}(D_{50},\bm{\theta}_2)$",
                  r"$\hat{L}(D_{500},\bm{\theta}_1)$",
                  r"$\hat{L}(D_{500},\bm{\theta}_2)$"
                  ])

plt.axvline(Ltest0, 0, 1.0, color=matplotlib.colormaps["tab10"](0), label=r"$L(\bm{\theta}_1)$", linewidth = 3)
plt.axvline(Ltest1, 0, 1.0, color=matplotlib.colormaps["tab10"](1), label=r"$L(\bm{\theta}_2)$", linewidth = 3)

plt.legend(prop={'size': 26})
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.ylabel("", fontsize=26)
plt.savefig("Lhat.pdf",  format = "pdf",bbox_inches='tight')
plt.clf()

jet = matplotlib.colormaps["Set2"]
colors = [jet(0), jet(1), jet(2), jet(3)]

positive_batch_losses_50 = [
    Ltest0 - batch_losses_50_0[batch_losses_50_0 < Ltest0] ,
    Ltest1 - batch_losses_50_1[batch_losses_50_1 < Ltest1],
]
positive_batch_losses_500 = [
    Ltest0 - batch_losses_500_0[batch_losses_500_0 < Ltest0] ,
    Ltest1 - batch_losses_500_1[batch_losses_500_1 < Ltest1],
]

plt.hist([*positive_batch_losses_50, *positive_batch_losses_500],
         density=True,
         align="mid",
         bins=10,
         color = colors,
         label = [r"$L(\bm{\theta}_1) - \hat{L}(D_{50},\bm{\theta}_1)$",
                  r"$L(\bm{\theta}_2) - \hat{L}(D_{50},\bm{\theta}_2)$",
                  r"$L(\bm{\theta}_1) - \hat{L}(D_{500},\bm{\theta}_1)$",
                  r"$L(\bm{\theta}_2) - \hat{L}(D_{500},\bm{\theta}_2)$",
                  ])

plt.legend(prop={'size': 26})
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.ylabel("", fontsize=26)
plt.savefig("Lhat_positive.pdf",  format = "pdf",bbox_inches='tight')
plt.clf()



rates_50_0 = np.loadtxt("Fig2/rates/50-Models0.txt")
rates_500_0 = np.loadtxt("Fig2/rates/500-Models0.txt")
rates_50_1 = np.loadtxt("Fig2/rates/50-Models1.txt")
rates_500_1 = np.loadtxt("Fig2/rates/500-Models1.txt")

print(rates_50_0.shape)

x = np.linspace(0, 0.1, 30)

plt.hist([rates_50_0[:, 0], rates_50_1[:, 0]],
         density=True,
         bins = x,
         align="left",
         color = colors[:2],
         label = [r"$\alpha(\bm{\theta}_0,D_{50})$",
                  r"$\alpha(\bm{\theta}_1,D_{50})$",
                  ]
         )
x = np.linspace(0, 0.1, 100)
y = 50*np.exp(-x*50)

plt.plot(x, y, color='red', label = "Exp(50) Density", linewidth = 3)
plt.legend()
plt.savefig(f"alpha_density_50.pdf",  format = "pdf",bbox_inches='tight')
plt.clf()

x = np.linspace(0, 0.01, 30)

plt.hist([rates_500_0[:, 0], rates_500_1[:, 0]],
         density=True,
         bins = x,
         align="left",
         color = colors[:2],
         label = [r"$\alpha(\bm{\theta}_0,D_{500})$",
                  r"$\alpha(\bm{\theta}_1,D_{500})$",
                  ]
         )
x = np.linspace(0, 0.01, 100)
y = 500*np.exp(-x*500)

plt.plot(x, y, color='red', label = "Exp(500) Density", linewidth = 3)
plt.legend()
plt.savefig(f"alpha_density_500.pdf",  format = "pdf",bbox_inches='tight')
plt.clf()