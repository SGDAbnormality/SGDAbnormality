import pickle
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib

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


network = "inception"
#network = "MLP"
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
batches = [250, 500, 5000, 5001]
wd = [0.0, 0.0, 0.0, 0.0]

#batches = [250, 500, 5000]
#skips = [0.0, 0.001, 0.005, 0.01]


for idx, batch in enumerate(batches):
    name_model = f"{network}_{dataset_name}_{TRAIN_SIZE}_{TEST_SIZE}_{batch}_{wd[idx]}"

    if batch==5001:
        tmp_model = f"{network}_{dataset_name}_{TRAIN_SIZE}_{TEST_SIZE}_{500}_{0.1}"
        with open(f'./metrics/metrics_{tmp_model}.pickle', 'rb') as f:
            metrics=pickle.load(f)
    else:
        with open(f'./metrics/metrics_{name_model}.pickle', 'rb') as f:
            metrics=pickle.load(f)

    for key in metrics.keys():
        metrics[key]=np.array(metrics[key])

        #if batch == 5001:
        #    metrics[key] = metrics[key][:82]
    all_metrics[str(batch)]=metrics


for batch in batches:
    all_metrics[str(batch)]['gen']=all_metrics[str(batch)]['L_test']-all_metrics[str(batch)]['L_train']

for batch in batches:
    all_metrics[str(batch)]['Iinv_alpha_mean'] = all_metrics[str(batch)]['Iinv_alpha_mean'].reshape(-1)

# for metric in all_metrics[str(batches[0])].keys():
#     all_metrics['5001'][metric][0] = all_metrics[str("5000")][metric][0]

jet = plt.get_cmap('Set2')
fig = plt.figure(figsize=(16, 8))
axis = plt.gca()
for i, batch in enumerate(batches):
    metric = 'L_test'
    label = "Batch Size " + str(batch) if batch != 5001 else r"Batch Size 500 $\ell_2$"
    axis.plot(all_metrics[str(batch)]['iter'], 
             all_metrics[str(batch)][metric], 
             label=label,
             linewidth=8,
             color=jet(i))
    metric = 'L_train'
    axis.plot(all_metrics[str(batch)]['iter'], 
             all_metrics[str(batch)][metric], 
             linewidth=8,
             alpha = 0.5,
            color=jet(i))
#plt.legend(ncols= 2)
axis.set_xlabel('Iterations')
axis.grid()
plt.savefig(f'./figs/{network}/L_Lhat.pdf', bbox_inches='tight', format='pdf', dpi=300)
#plt.cla()
lines, labels = axis.get_legend_handles_labels()
legend = plt.legend(lines, labels, loc=0, ncol=4, fancybox=True, shadow=True)
for line in legend.get_lines():
    line.set_linewidth(20)
def export_legend(legend, filename=f"./figs/{network}/legend2.pdf"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

export_legend(legend)

#plt.show()
plt.clf()



fig = plt.figure(figsize=(16, 8))
metric = 'var_test'
for i, batch in enumerate(batches):

    plt.plot(all_metrics[str(batch)]['iter'], 
             all_metrics[str(batch)][metric], 
             label="Batch Size " + str(batch) if batch != 5001 else "Batch Size 5000 L2",
                linewidth=8,
                color=jet(i))
plt.xlabel('Iterations')
plt.grid()
plt.savefig(f'./figs/{network}/variance.pdf', bbox_inches='tight', format='pdf', dpi=300)
plt.clf()

fig = plt.figure(figsize=(16, 8))
metric = 'std_loss'
for i, batch in enumerate(batches):

    plt.plot(all_metrics[str(batch)]['iter'], 
             np.sqrt(all_metrics[str(batch)]['var_test']), 
             label="Batch Size " + str(batch) if batch != 5001 else "Batch Size 5000 L2",
                linewidth=8,
                color=jet(i))
plt.xlabel('Iterations')
#plt.ylim(-0.1,2.5)
plt.grid()
plt.savefig(f'./figs/{network}/{metric}.pdf', bbox_inches='tight', format='pdf', dpi=300)
plt.clf()


fig = plt.figure(figsize=(16, 8))
metric = 'mean_alpha'
for i, batch in enumerate(batches):

    plt.plot(all_metrics[str(batch)]['iter'], 
             all_metrics[str(batch)][metric], 
             label="Batch Size " + str(batch) if batch != 5001 else "Batch Size 5000 L2",
                linewidth=8,
                color=jet(i))
plt.xlabel('Iterations')
plt.grid()
plt.savefig(f'./figs/{network}/mean_alpha.pdf', bbox_inches='tight', format='pdf', dpi=300)
plt.clf()


fig = plt.figure(figsize=(16, 8))
metric = 'var_alpha'
for i, batch in enumerate(batches):

    plt.plot(all_metrics[str(batch)]['iter'], 
             all_metrics[str(batch)][metric], 
             label="Batch Size " + str(batch) if batch != 5001 else "Batch Size 5000 L2",
                linewidth=8,
                color=jet(i))
plt.xlabel('Iterations')
#plt.yscale('log')  # Use log scale for better visualization of decay
plt.grid()
plt.savefig(f'./figs/{network}/var_alpha.pdf', bbox_inches='tight', format='pdf', dpi=300)
plt.clf()


fig = plt.figure(figsize=(16, 8))
metric = 'variance_mean_term'
for i, batch in enumerate(batches):
    val = 0.5*np.power(2*all_metrics[str(batch)]['mean_alpha'],-3.0/2.0)*all_metrics[str(batch)]['var_alpha']
    plt.plot(all_metrics[str(batch)]['iter'], 
             val, 
             label=f"Batch Size {batch}",
             linewidth=8,
             color=jet(i))

plt.xlabel('Iterations', fontsize=fontsize-10)
plt.grid()
plt.yscale('log')  # Use log scale for better visualization of decay
plt.savefig(f'./figs/{network}/{metric}.pdf', bbox_inches='tight', format='pdf', dpi=300)


fig = plt.figure(figsize=(16, 8))
metric = 'bias_term'
for i, batch in enumerate(batches):
    val = 0.5*np.power(2*all_metrics[str(batch)]['mean_alpha'],-3.0/2.0)*all_metrics[str(batch)]['var_alpha']*all_metrics[str(batch)]['var_test']
    plt.plot(all_metrics[str(batch)]['iter'], 
             val, 
             label=f"Batch Size {batch}",
             linewidth=8,
             color=jet(i),
             alpha=0.9)

plt.xlabel('Iterations', fontsize=fontsize-10)
plt.grid()

plt.savefig(f'./figs/{network}/bias_term.pdf', bbox_inches='tight', format='pdf', dpi=300)
plt.yscale('log')  # Use log scale for better visualization of decay
plt.savefig(f'./figs/{network}/logbias_term.pdf', bbox_inches='tight', format='pdf', dpi=300)


plt.clf()





fig = plt.figure(figsize=(16, 8))
metric = 'Iinv_alpha_mean'
for i, batch in enumerate(batches):

    plt.plot(all_metrics[str(batch)]['iter'], 
             all_metrics[str(batch)][metric], 
             label="Batch Size " + str(batch) if batch != 5001 else "Batch Size 5000 L2",
                linewidth=8,
                color=jet(i))
plt.xlabel('Iterations')
plt.grid()
plt.savefig(f'./figs/{network}/{metric}.pdf', bbox_inches='tight', format='pdf', dpi=300)
plt.clf()





fig = plt.figure(figsize=(16, 8))
metric = 'Approximation_Error'
for i, batch in enumerate(batches):
    bias_term = 0.5*np.power(2*all_metrics[str(batch)]['mean_alpha'],-3.0/2.0)*all_metrics[str(batch)]['var_alpha']*np.sqrt(all_metrics[str(batch)]['var_test'])
    val = all_metrics[str(batch)]['L_test']-all_metrics[str(batch)]['Iinv_alpha_mean']+bias_term
    plt.plot(all_metrics[str(batch)]['iter'], 
             val, 
             label=f"Lhat Batch Size {batch}",
             linewidth=8,
             color=jet(i),
             alpha=1.0,
             linestyle='dotted')  # Added dotted line style

    plt.plot(all_metrics[str(batch)]['iter'], 
             all_metrics[str(batch)]['L_train'], 
             label=f"Lhat Approx Batch Size {batch}",
             linewidth=8,
             color=jet(i),
             alpha=0.5)
    

plt.xlabel('Iterations')
plt.grid()
#plt.legend(loc='upper right')
plt.savefig(f'./figs/{network}/{metric}.pdf', bbox_inches='tight', format='pdf', dpi=300)
plt.clf()



fig = plt.figure(figsize=(16, 8))
metric = 'Approximation_Error_Var'
for i, batch in enumerate(batches):
    bias_term = 0.5*np.power(2*all_metrics[str(batch)]['mean_alpha'],-3.0/2.0)*all_metrics[str(batch)]['var_alpha']*np.sqrt(all_metrics[str(batch)]['var_test'])
    var_term = np.sqrt(2*all_metrics[str(batch)]['mean_alpha'])*np.sqrt(all_metrics[str(batch)]['var_test'])
    val = all_metrics[str(batch)]['L_test']-var_term+bias_term
    plt.plot(all_metrics[str(batch)]['iter'], 
             val, 
             label=f"Batch Size {batch}",
             linewidth=8,
             color=jet(i),
             alpha=1.0,
             linestyle='dotted')  # Added dotted line style

    plt.plot(all_metrics[str(batch)]['iter'], 
             all_metrics[str(batch)]['L_train'], 
             label=f"Batch Size {batch}",
             linewidth=8,
             color=jet(i),
             alpha=0.5)
    
plt.ylim(-.01, 2.1)
plt.xlabel('Iterations')
plt.grid()
plt.savefig(f'./figs/{network}/{metric}.pdf', bbox_inches='tight', format='pdf', dpi=300)
plt.clf()




fig = plt.figure(figsize=(16, 8))
metric = 'var_alpha_loss'
for i, batch in enumerate(batches):
    val = all_metrics[str(batch)]['var_alpha']*all_metrics[str(batch)]['var_test']
    plt.plot(all_metrics[str(batch)]['iter'], 
             val, 
             label="Batch Size " + str(batch) if batch != 5001 else "Batch Size 5000 L2",
                linewidth=8,
                color=jet(i))

    
plt.xlabel('Iterations')
plt.grid()
plt.savefig(f'./figs/{network}/{metric}.pdf', bbox_inches='tight', format='pdf', dpi=300)
plt.clf()


fig = plt.figure(figsize=(16, 8))
metric = 'phi_alpha'
for i, batch in enumerate(batches):
    val = np.sqrt(2*all_metrics[str(batch)]['mean_alpha'])  - 0.5*np.power(2*all_metrics[str(batch)]['mean_alpha'],-3.0/2.0)*all_metrics[str(batch)]['var_alpha']
    plt.plot(all_metrics[str(batch)]['iter'], 
             val, 
             label="Batch Size " + str(batch) if batch != 5001 else "Batch Size 5000 L2",
                linewidth=8,
                color=jet(i))

plt.ylim(-.05, 1.5)
plt.xlabel('Iterations')
plt.grid()
plt.savefig(f'./figs/{network}/{metric}.pdf', bbox_inches='tight', format='pdf', dpi=300)
plt.clf()