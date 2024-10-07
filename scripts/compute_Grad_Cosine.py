import os
import argparse
import pickle
from itertools import combinations

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib
from scipy.spatial import distance
from tqdm import tqdm

plt.rcParams.update({
    'font.size': 10,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}\usepackage{bm}'
})
mpl.rc('font',family='Times New Roman')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.figsize'] = (16, 9)
fontsize = 46
matplotlib.rcParams.update({'font.size': fontsize})


dataset_name = "CIFAR10"



parser = argparse.ArgumentParser()
parser.add_argument("-m", "--MODEL", help="MODEL", default= "MLP", type=str)
parser.add_argument("-b", "--BATCH_SIZE", help="BATCH_SIZE", default= 2500, type=int)
parser.add_argument("-w", "--WEIGHT_DECAY", help="WEIGHT_DECAY", default= 0.0, type=float)


args = parser.parse_args()

TRAIN_SIZE = 0
TEST_SIZE = 0
network = args.MODEL
BATCH_SIZE = args.BATCH_SIZE
WEIGHT_DECAY = args.WEIGHT_DECAY

all_metrics = {}
#gradient_types = [
#    "L", "Lhat", "Lbatch", "Iinv_D", "Iinv_batch", "alpha_D", "alpha_batch"
#]

gradient_types_batch = [
    "Lbatch", "L", "Iinv_batch", "alpha_batch"
]

gradient_types_D = [
    "Lhat", "L", "Iinv_D", "alpha_D"
]


name_model = f"{network}_{dataset_name}_{TRAIN_SIZE}_{TEST_SIZE}_{BATCH_SIZE}_{WEIGHT_DECAY}"

with open(f'./metrics/metrics_{name_model}.pickle', 'rb') as f:
    metrics = pickle.load(f)

for key in metrics.keys():
    metrics[key] = np.array(metrics[key])

grad_directory_path = f'./gradients/{name_model}/'

tq = tqdm(metrics['iter'])

for grad1, grad2 in combinations(gradient_types_batch, 2):
    metrics[f"cosine_{grad1}_{grad2}"]=[]


for grad1, grad2 in combinations(gradient_types_D, 2):
    metrics[f"cosine_{grad1}_{grad2}"]=[]

for grad1, grad2 in combinations(gradient_types_batch, 2):
    metrics[f"distance_{grad1}_{grad2}"]=[]

for grad1, grad2 in combinations(gradient_types_D, 2):
    metrics[f"distance_{grad1}_{grad2}"]=[]

for grad_type in gradient_types_D:
    metrics[f"cosine_Lbatch_{grad_type}"] = []

for grad_type in gradient_types_batch:
    if grad_type == "Lbatch":
        continue  # Skip the gradient of L projecting onto itself
    metrics[f"norm_projection_{grad_type}_onto_Lbatch"]=[]

for grad_type in gradient_types_D:
    if grad_type == "Lhat":
        continue  # Skip the gradient of L projecting onto itself
    metrics[f"norm_projection_{grad_type}_onto_Lhat"]=[]

tq = tqdm(metrics['iter'])
for it in tq:

    # Dictionary to store the loaded gradients
    gradients_batch = {}

    # Loop through each gradient type and load the corresponding pickle file
    for grad_type in gradient_types_batch:
        file_path = os.path.join(grad_directory_path, f"gradients_{grad_type}{int(it)}.pickle")

        with open(file_path, "rb") as handle:
            gradients_batch[grad_type] = pickle.load(handle)

    # Dictionary to store the loaded gradients
    gradients_D = {}
    for grad_type in gradient_types_D:
        file_path = os.path.join(grad_directory_path, f"gradients_{grad_type}{int(it)}.pickle")

        with open(file_path, "rb") as handle:
            gradients_D[grad_type] = pickle.load(handle)

    for grad1, grad2 in combinations(gradient_types_batch, 2):
        cosine_dist = 1 - distance.cosine(gradients_batch[grad1], gradients_batch[grad2])

        # Store the result in the metrics dictionary with the pair as the key
        metrics[f"cosine_{grad1}_{grad2}"].append(cosine_dist)

    for grad1, grad2 in combinations(gradient_types_D, 2):
        cosine_dist = 1 - distance.cosine(gradients_D[grad1], gradients_D[grad2])

        # Store the result in the metrics dictionary with the pair as the key
        metrics[f"cosine_{grad1}_{grad2}"].append(cosine_dist)

    for grad1, grad2 in combinations(gradient_types_batch, 2):
        metrics[f"distance_{grad1}_{grad2}"].append(np.linalg.norm(gradients_batch[grad1] - gradients_batch[grad2]))

    for grad1, grad2 in combinations(gradient_types_D, 2):
        metrics[f"distance_{grad1}_{grad2}"].append(np.linalg.norm(gradients_D[grad1] - gradients_D[grad2]))

    # Get gradient of L
    grad_Lbatch = gradients_batch["Lbatch"]

    for grad_type in gradient_types_D:
        cosine_dist = 1 - distance.cosine(grad_Lbatch, gradients_D[grad_type])
        metrics[f"cosine_Lbatch_{grad_type}"].append(cosine_dist)

    # Loop through all gradients and compute their projection onto grad_L
    for grad_type, grad_vec in gradients_batch.items():
        if grad_type == "Lbatch":
            continue  # Skip the gradient of L projecting onto itself

        # Compute projection of grad_vec onto grad_L
        projection = (np.dot(grad_vec, grad_Lbatch) / np.dot(grad_Lbatch, grad_Lbatch)) * grad_Lbatch

        metrics[f"norm_projection_{grad_type}_onto_Lbatch"].append(np.linalg.norm(projection))



    # Get gradient of L
    grad_Lhat= gradients_D["Lhat"]

    # Loop through all gradients and compute their projection onto grad_L
    for grad_type, grad_vec in gradients_D.items():
        if grad_type == "Lhat":
            continue  # Skip the gradient of L projecting onto itself

        # Compute projection of grad_vec onto grad_L
        projection = (np.dot(grad_vec, grad_Lhat) / np.dot(grad_Lhat, grad_Lhat)) * grad_Lhat

        metrics[f"norm_projection_{grad_type}_onto_Lhat"].append(np.linalg.norm(projection))

with open(f'metrics/metrics_{name_model}.pickle', 'wb') as f:
    pickle.dump(metrics, f, pickle.HIGHEST_PROTOCOL)

