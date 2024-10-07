import pickle
import numpy as np
from matplotlib import pyplot as plt



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
WEIGHT_DECAY = 0.01




all_metrics = {}
batches = [250, 500, 2500, 5000]
#batches = [50]


for batch in batches:
    name_model = f"{network}_{dataset_name}_{TRAIN_SIZE}_{TEST_SIZE}_{batch}"


    with open(f'./FigGradientAligment/metrics/L_metrics_{name_model}.pickle', 'rb') as f:
        metrics=pickle.load(f)

    for key in metrics.keys():
        metrics[key]=np.array(metrics[key])

    all_metrics[str(batch)]=metrics

for batch in batches:
    all_metrics[str(batch)]['gen']=all_metrics[str(batch)]['L']-all_metrics[str(batch)]['Lhat']


#for batch in batches:
#        all_metrics[str(batch)]['iter']=all_metrics[str(batch)]['iter']*(batch/50000)

metric = 'gen'
for batch in batches:
    plt.plot(all_metrics[str(batch)]['iter'], all_metrics[str(batch)][metric], label=metric+"_"+str(batch))
plt.legend()
plt.show()


for batch in batches:
    metric = 'Lhat'
    plt.plot(all_metrics[str(batch)]['iter'], all_metrics[str(batch)][metric], label=metric+"_"+str(batch))
    metric = 'L'
    plt.plot(all_metrics[str(batch)]['iter'], all_metrics[str(batch)][metric], label=metric + "_" + str(batch))

plt.legend()
plt.show()



for batch in batches:
    print(f"Batch:{batch} - {np.min(all_metrics[str(batch)]['L'])} - {all_metrics[str(batch)]['iter'][np.argmin(all_metrics[str(batch)]['L'])]}")