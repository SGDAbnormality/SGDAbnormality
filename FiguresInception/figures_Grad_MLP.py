import pickle
import numpy as np
from matplotlib import pyplot as plt

from utils import find_change_point



network = "Inception_SGD"
dataset_name = "CIFAR10"
TRAIN_SIZE = 0
TEST_SIZE = 0
# Hyper-Parameters
RANDOM_SEED = 2147483647
LEARNING_RATE = 0.01
N_ITERS = 2000000
IMG_SIZE = 32
N_CLASSES = 10
WEIGHT_DECAY = 0.0




all_metrics = {}
batches = [50, 250, 500, 2500, 5000]
scales = {'Inception_SGD': 1.0, 'MLP':5.0/3.0, 'INCEPTION': 1.0}
#batches = [50, 500, 5000]
#scales = {'50': 3.5,'500':0.8, '5000':0.6}

networks = ['Inception_SGD', 'MLP']
wd = {'Inception_SGD': 0.0, 'MLP': 0.0, 'INCEPTION': 0.02}

batch=250
for network in networks:
    name_model = f"{network}_{dataset_name}_{TRAIN_SIZE}_{TEST_SIZE}_{batch}_{wd[network]}"


    with open(f'../metrics/metrics_{name_model}.pickle', 'rb') as f:
        metrics=pickle.load(f)

    for key in metrics.keys():
        metrics[key]=np.array(metrics[key])

    all_metrics[network]=metrics

for network in networks:
    all_metrics[network]['gen']=all_metrics[network]['L']-all_metrics[network]['Lhat']

for network in networks:
    all_metrics[network]['gen_batch']=all_metrics[network]['L']-all_metrics[network]['Lbatch']


for network in networks:
        all_metrics[network]['iter']=all_metrics[network]['iter']/scales[network]


for network in networks:
    metric = 'L'
    plt.plot(all_metrics[network]['iter'], all_metrics[network][metric], label=metric+"_"+network)
    metric = 'Lhat'
    plt.plot(all_metrics[network]['iter'], all_metrics[network][metric], label=metric + "_" +network)
plt.xlim(0,3000)
plt.legend()
plt.show()


for network in networks:
    metrics = all_metrics[network]
    plt.scatter(metrics['iter'],metrics['cosine_Iinv'], label='Cosine Inv '+network)
    plt.scatter(metrics['iter'],metrics['cosine_Alpha'], label='Cosine Alpha '+network)
    #plt.scatter(metrics['iter'], np.sign(np.multiply(metrics['gen'],metrics['gen_batch'])), label='sign ' +network)
    plt.ylim(-1.1, 1.1)
    plt.legend()
    plt.show()



metric = 'cosine_Alpha'
for network in networks:
    plt.hist(all_metrics[network][metric], label=network, density=True)
plt.xlim(-0.5,1)
plt.legend()
plt.show()


metric = 'cosine_Lhat_Lbatch'
for network in networks:
    plt.hist(all_metrics[network][metric], label=network, density=True)
plt.legend()
plt.show()



metrics = {'cosine_Lhat_Lbatch', 'cosine_Iinv', 'cosine_Alpha'}

for metric in metrics:
    data = []

    for network in networks:
        ratio = all_metrics[network][metric]
        data.append(ratio)

    plt.boxplot(data, labels=[network for network in networks])
    plt.ylim(-1.1,1.1)
    plt.ylabel(metric)
    plt.xlabel('Batches')
    plt.title('Boxplot of Gradient Ratios Across Batches')
    plt.show()

metric = 'variance'
for network in networks:
    plt.plot(all_metrics[network]['iter'], all_metrics[network][metric], label=metric+"_"+network)
#plt.ylim(-1,1)
plt.xlim(0,3000)
plt.legend()
plt.show()


metric = 'rate_D'
for network in networks:
    plt.plot(all_metrics[network]['iter'], all_metrics[network][metric], label=metric+"_"+network)
plt.xlim(0,3000)
plt.ylim(0,1)
plt.legend()
plt.show()


metric = 'rate_B'
for network in networks:
    plt.plot(all_metrics[network]['iter'], all_metrics[network][metric], label=metric+"_"+network)
plt.xlim(0,3000)
plt.legend()
plt.show()



metric = 'gen_batch'
for network in networks:
    plt.plot(all_metrics[network]['iter'], all_metrics[network][metric], label=metric+"_"+network)

plt.xlim(0,3000)
#plt.ylim(0,3)
plt.legend()
plt.show()


metric = 'gradients_Iinv_batch_norm'
for network in networks:
    plt.plot(all_metrics[network]['iter'], all_metrics[network]['gradients_Iinv_batch_norm']/all_metrics[network]['gradients_Lbatch_norm'], label=metric+"_"+network)
plt.ylim(0,10)
plt.xlim(0,3000)
plt.legend()
plt.show()


metric = 'gradients_Iinv_batch_norm'
for network in networks:
    plt.hist(all_metrics[network]['gradients_Iinv_batch_norm']/all_metrics[network]['gradients_Lbatch_norm'], label=network, density=True)
#plt.xlim(-0.5,1)
plt.legend()
plt.show()


metric = 'gradients_Iinv_batch_norm'
data = []

for network in networks:
    ratio = all_metrics[network]['gradients_Iinv_batch_norm'] / all_metrics[network]['gradients_Lbatch_norm']
    data.append(ratio)

plt.boxplot(data, labels=[network for batch in batches])
plt.ylabel('Ratio of gradients_Iinv_batch_norm to gradients_Lbatch_norm')
plt.xlabel('Batches')
plt.title('Boxplot of Gradient Ratios Across Batches')
#plt.ylim(0,10)
plt.show()


metric = 'normalized_ration'
data = []

for network in networks:
    ratio = all_metrics[network]['gradients_Iinv_batch_norm'] / np.sqrt(2*np.abs(all_metrics[network]['rate_B']))
    data.append(ratio)

plt.boxplot(data, labels=[network for batch in batches])
plt.ylabel('Ratio of gradients_Iinv_batch_norm to gradients_Lbatch_norm')
plt.xlabel('Batches')
plt.title('Boxplot of Gradient Ratios Across Batches')
plt.show()












for network in networks:
    print(np.corrcoef(all_metrics[network]['inv_B'],all_metrics[network]['inv_B_approx'])[0,1])




