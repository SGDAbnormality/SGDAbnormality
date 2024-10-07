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


network = "INCEPTION"
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
batches = [250, 500, 5000]

#all_metrics = {}
#batches = [250, 5001]

scales = {'50': 1, '250':1, '500':1, '5001':1, '5000':1}
#scales = {'50': 3.5,'500':0.8, '5000':0.6}


for batch in batches:
    name_model = f"{network}_{dataset_name}_{TRAIN_SIZE}_{TEST_SIZE}_{batch}_{WEIGHT_DECAY}"

    if batch==5001:
        tmp_model = f"{network}_{dataset_name}_{TRAIN_SIZE}_{TEST_SIZE}_{5000}_{0.05}"
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
    all_metrics[str(batch)]['gen']=all_metrics[str(batch)]['L']-all_metrics[str(batch)]['Lhat']

for batch in batches:
    all_metrics[str(batch)]['gen_batch']=all_metrics[str(batch)]['L']-all_metrics[str(batch)]['Lbatch']

for batch in batches:
    all_metrics[str(batch)]['iter']=all_metrics[str(batch)]['iter']/scales[str(batch)]

# for metric in all_metrics[str(batches[0])].keys():
#     all_metrics['5001'][metric][0] = all_metrics[str("5000")][metric][0]

jet = plt.get_cmap('Set2')
fig = plt.figure(figsize=(16, 8))
axis = plt.gca()
for i, batch in enumerate(batches):
    metric = 'L'
    label = "Batch Size " + str(batch) if batch != 5001 else r"Batch Size 5000 $\ell_2$"
    axis.plot(all_metrics[str(batch)]['iter'], 
             all_metrics[str(batch)][metric], 
             label=label,
             linewidth=8,
             color=jet(i))
    metric = 'Lhat'
    axis.plot(all_metrics[str(batch)]['iter'], 
             all_metrics[str(batch)][metric], 
             linewidth=8,
             alpha = 0.5,
            color=jet(i))
#plt.legend(ncols= 2)
axis.set_xlabel('Iterations')
axis.grid()
plt.savefig('L_Lhat.pdf', bbox_inches='tight', format='pdf', dpi=300)
plt.show()
plt.cla()
lines, labels = axis.get_legend_handles_labels()
legend = plt.legend(lines, labels, loc=0, ncol=4, fancybox=True, shadow=True)
for line in legend.get_lines():
    line.set_linewidth(20)
def export_legend(legend, filename="legend2.pdf"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

export_legend(legend)

plt.clf()


fig = plt.figure(figsize=(16, 8))
jet = plt.get_cmap('Set2')
metric = 'cosine_Alpha'
plt.hist(
    [all_metrics[str(batch)][metric] for batch in batches], 
    bins = [-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1],
    label=["Batch Size " + str(batch) if batch != 5001 else "Batch Size 5000 L2" for batch in batches], 
    density=True, 
    align='right',
    color = [jet(i) for i in range(len(batches))])
plt.xlabel('Cosine Similarity')
plt.savefig('cosine_Alpha.pdf', bbox_inches='tight', format='pdf', dpi=300)
plt.clf()

fig = plt.figure(figsize=(16, 8))
metric = 'cosine_Lhat_Lbatch'
plt.hist(
    [all_metrics[str(batch)][metric] for batch in batches], 
    bins = [-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1],
    label=["Batch Size " + str(batch) if batch != 5001 else "Batch Size 5000 L2" for batch in batches], 
    density=True, 
    align='right',
    color = [jet(i) for i in range(len(batches))])
plt.xlabel('Cosine Similarity')
plt.savefig('cosine_Lhat_Lbatch.pdf', bbox_inches='tight', format='pdf', dpi=300)
plt.clf()


fig = plt.figure(figsize=(16, 8))
metric = 'cosine_Iinv'
plt.hist(
    [all_metrics[str(batch)][metric] for batch in batches], 
    bins =  [-1.1, -0.9, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.9, 1.1],
    label=["Batch Size " + str(batch) if batch != 5001 else "Batch Size 5000 L2" for batch in batches], 
    density=True, 
    align='mid',
    color = [jet(i) for i in range(len(batches))])
plt.xlabel('Cosine Similarity')
plt.savefig('cosine_Iinv.pdf', bbox_inches='tight', format='pdf', dpi=300)
plt.clf()



fig = plt.figure(figsize=(16, 8))
metric = 'variance'
for i, batch in enumerate(batches):

    plt.plot(all_metrics[str(batch)]['iter'], 
             all_metrics[str(batch)][metric], 
             label="Batch Size " + str(batch) if batch != 5001 else "Batch Size 5000 L2",
                linewidth=8,
                color=jet(i))
plt.xlabel('Iterations')
plt.grid()
plt.savefig('variance.pdf', bbox_inches='tight', format='pdf', dpi=300)
plt.clf()



fig = plt.figure(figsize=(16, 8))
metric = 'rate_D'
for i, batch in enumerate(batches):
    plt.plot(all_metrics[str(batch)]['iter'], 
             all_metrics[str(batch)][metric], 
             label="Batch Size " + str(batch) if batch != 5001 else "Batch Size 5000 L2",
                linewidth=8,
                color=jet(i))
    plt.plot(all_metrics[str(batch)]['iter'], 
             all_metrics[str(batch)]["rate_B"], 
                linewidth=8,
                alpha = 0.5,
                color=jet(i))
plt.xlabel('Iterations')
plt.grid()
plt.savefig('rate_D.pdf', bbox_inches='tight', format='pdf', dpi=300)
plt.show()
plt.clf()



fig = plt.figure(figsize=(16, 8))
metric = 'rate_B'
for i, batch in enumerate(batches):
    plt.plot(all_metrics[str(batch)]['iter'], 
             all_metrics[str(batch)][metric], 
             label="Batch Size " + str(batch) if batch != 5001 else "Batch Size 5000 L2",
                linewidth=8,
                color=jet(i))
plt.xlabel('Iterations')
plt.grid()
plt.savefig('rate_B.pdf', bbox_inches='tight', format='pdf', dpi=300)
plt.show()
plt.clf()


fig = plt.figure(figsize=(16, 8))
metric = 'gen_batch'
for i, batch in enumerate(batches):
    plt.plot(all_metrics[str(batch)]['iter'], 
             all_metrics[str(batch)][metric], 
             label="Batch Size " + str(batch) if batch != 5001 else "Batch Size 5000 L2",
                linewidth=8,
                color=jet(i))
plt.xlabel('Iterations')
plt.grid()
plt.savefig('gen_batch.pdf', bbox_inches='tight', format='pdf', dpi=300)
plt.show()
plt.clf()


# fig = plt.figure(figsize=(16, 8))
# metric = 'lamb_rate_D'
# for i, batch in enumerate(batches):
#     plt.plot(all_metrics[str(batch)]['iter'],
#              all_metrics[str(batch)][metric],
#              label="Batch Size " + str(batch) if batch != 5001 else "Batch Size 5000 L2",
#                 linewidth=8,
#                 color=jet(i))
# plt.xlabel('Iterations')
# plt.ylim(0,1.1)
# plt.grid()
# plt.savefig('lamb_rate_D.pdf', bbox_inches='tight', format='pdf', dpi=300)
# plt.show()
# plt.clf()


# fig = plt.figure(figsize=(16, 8))
# for i, batch in enumerate(batches):
#     lamb = all_metrics[str(batch)]['lamb_rate_B']*np.sign(all_metrics[str(batch)]['gen_batch'])
#     J = all_metrics[str(batch)]['J_B']
#     plt.plot(all_metrics[str(batch)]['iter'],
#              np.exp(-J+all_metrics[str(batch)]['L']),
#              label="1",
#                 linewidth=8,
#                 color=jet(0))
#     plt.plot(all_metrics[str(batch)]['iter'],
#              lamb * np.exp(-J)/2,
#              label="2",
#                 linewidth=8,
#                 color=jet(1))
#     plt.plot(all_metrics[str(batch)]['iter'],
#              lamb**2 * np.exp(-J)/6,
#              label="3",
#                 linewidth=8,
#                 color=jet(2))
#     plt.plot(all_metrics[str(batch)]['iter'],
#              lamb**3 * np.exp(-J)/24,
#              label="4",
#                 linewidth=8,
#                 color=jet(3))
#     plt.title(str(batch))
#     plt.xlabel('Iterations')
#     plt.grid()
#     plt.legend(loc='best')
#     plt.savefig('lamb_rate_D.pdf', bbox_inches='tight', format='pdf', dpi=300)
#     plt.show()
#     plt.clf()

# fig = plt.figure(figsize=(16, 8))
# for i, batch in enumerate(batches):
#     lamb = all_metrics[str(batch)]['lamb_rate_D']
#     J = all_metrics[str(batch)]['J_D']
#     factor = 1.0#all_metrics[str(batch)]['L'][-1]/np.cumsum(np.exp(-J))[-1]
#     plt.plot(all_metrics[str(batch)]['iter'],
#              np.cumsum(np.exp(-J))*factor,
#              label="1",
#                 linewidth=8,
#                 color=jet(0))
#     plt.plot(all_metrics[str(batch)]['iter'],
#              np.cumsum(lamb * np.exp(-J)/2)*factor,
#              label="2",
#                 linewidth=8,
#                 color=jet(1))
#     plt.plot(all_metrics[str(batch)]['iter'],
#              np.cumsum(lamb**2 * np.exp(-J)/6)*factor,
#              label="3",
#                 linewidth=8,
#                 color=jet(2))
#     plt.plot(all_metrics[str(batch)]['iter'],
#              np.cumsum(lamb**3 * np.exp(-J)/24)*factor,
#              label="4",
#                 linewidth=8,
#                 color=jet(3))
#     plt.title(str(batch))
#     plt.ylim(0,1.1)
#     plt.xlabel('Iterations')
#     plt.grid()
#     plt.legend(loc='best')
#     plt.savefig('lamb_rate_D.pdf', bbox_inches='tight', format='pdf', dpi=300)
#     plt.show()
#     plt.clf()



# fig = plt.figure(figsize=(16, 8))
# for i, batch in enumerate(batches):
#     lamb = all_metrics[str(batch)]['lamb_rate_D']
#     J = all_metrics[str(batch)]['J_D']
#     factor = 1.0#all_metrics[str(batch)]['L'][-1]/np.cumsum(np.exp(-J))[-1]
#     plt.plot(all_metrics[str(batch)]['iter'],
#              np.cumsum(np.exp(-J))*factor,
#              label="1",
#                 linewidth=8,
#                 color=jet(0))
# plt.title(str(batch))
# plt.ylim(0,1.1)
# plt.xlabel('Iterations')
# plt.grid()
# plt.legend(loc='best')
# plt.savefig('lamb_rate_D.pdf', bbox_inches='tight', format='pdf', dpi=300)
# plt.show()
# plt.clf()

# fig = plt.figure(figsize=(16, 8))
# for i, batch in enumerate(batches):
#     lamb = all_metrics[str(batch)]['lamb_rate_B']
#     J = all_metrics[str(batch)]['J_B']
#     plt.plot(all_metrics[str(batch)]['iter'],
#              lamb * np.exp(-J)/2,
#              label=batch,
#                 linewidth=8,
#                 color=jet(i))
# plt.title("Var")
# plt.xlabel('Iterations')
# plt.grid()
# plt.legend(loc='best')
# plt.savefig('lamb_rate_D.pdf', bbox_inches='tight', format='pdf', dpi=300)
# plt.show()
# plt.clf()


# fig = plt.figure(figsize=(16, 8))
# metric = 'lamb_rate_B'
# for i, batch in enumerate(batches):
#     plt.plot(all_metrics[str(batch)]['iter'],
#              all_metrics[str(batch)][metric],
#              label="Batch Size " + str(batch) if batch != 5001 else "Batch Size 5000 L2",
#                 linewidth=8,
#                 color=jet(i))
# plt.xlabel('Iterations')
# plt.grid()
# plt.ylim(-2,20)
# plt.savefig('lamb_rate_D.pdf', bbox_inches='tight', format='pdf', dpi=300)
# plt.show()
# plt.clf()

# fig = plt.figure(figsize=(16, 8))
# metric = 'gen_batch'
# for i, batch in enumerate(batches):
#     plt.scatter(all_metrics[str(batch)][metric],
#                 all_metrics[str(batch)]['gradients_alpha_batch_norm'],
#              label="Batch Size " + str(batch) if batch != 5001 else "Batch Size 5000 L2",
#                 linewidth=8,
#                 color=jet(i))
# plt.xlabel('Iterations')
# plt.grid()
# plt.savefig('gen_batch.pdf', bbox_inches='tight', format='pdf', dpi=300)
# plt.show()
# plt.clf()

for batch in batches:
    fig = plt.figure(figsize=(16, 8))
    axis = plt.gca()
    axis.plot(all_metrics[str(batch)]['iter'],all_metrics[str(batch)]['inv_B_approx'], linewidth=8, color = jet(0),
                label = r"sign$(\alpha(B_t, \bm{\theta}_t))\sqrt{2 |\alpha(B_t, \bm{\theta}_t)|}\sigma(\bm{\theta}_t)$")
    axis.plot(all_metrics[str(batch)]['iter'],all_metrics[str(batch)]['L'], linewidth=8, color = jet(1),
                label = r"$L(\bm{\theta}_t)$")
    axis.plot(all_metrics[str(batch)]['iter'],all_metrics[str(batch)]['inv_B'], linewidth=8, color = jet(2),
                label = r"$\mathcal{I}^{-1}_{\bm{\theta}_t}(\alpha(B_t, \bm{\theta}_t))$")
    axis.set_ylim(-0.1, 2.3)
    axis.grid()
    axis.set_xlabel('Iterations')
    plt.savefig(f'approx_{batch}.pdf', bbox_inches='tight', format='pdf', dpi=300)
    plt.show()
    plt.cla()

plt.cla()
lines, labels = axis.get_legend_handles_labels()
legend = plt.legend(lines, labels, loc=0, ncol=4, fancybox=True, shadow=True)
for line in legend.get_lines():
    line.set_linewidth(20)
def export_legend(legend, filename="legend.pdf"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

export_legend(legend, "legend_approx.pdf")

plt.clf()



metrics = ['cosine_Lbatch_Iinv_D', 'cosine_Lbatch_alpha_D']

fig = plt.figure(figsize=(16, 8))
for metric in metrics:
    for i, batch in enumerate([250,5000]):
        vec = all_metrics[str(batch)][metric]
        plt.plot(all_metrics[str(batch)]['iter'],
                 vec,
                 label="Batch Size " + str(batch) if batch != 5001 else "Batch Size 5000 L2",
                    linewidth=8,
                    color=jet(i))

    plt.xlabel('Iterations')
    plt.grid()
    #plt.ylim(-0.5,10)
    #plt.xlim(0,50)
    plt.savefig(f'{metric}.pdf', bbox_inches='tight', format='pdf', dpi=300)
    plt.show()
    plt.clf()



metrics = ['J_D', 'distance_Lhat_L', 'cosine_Lhat_L']

fig = plt.figure(figsize=(16, 8))
for metric in metrics:
    for i, batch in enumerate(batches):
        vec = all_metrics[str(batch)][metric]
        plt.plot(all_metrics[str(batch)]['iter'],
                 vec,
                 label="Batch Size " + str(batch) if batch != 5001 else "Batch Size 5000 L2",
                    linewidth=8,
                    color=jet(i))

    plt.xlabel("Iterations")
    plt.grid()
    plt.savefig(f'{metric}.pdf', bbox_inches='tight', format='pdf', dpi=300)
    plt.clf()


metric = 'gradients_Iinv_D_norm'

for i, batch in enumerate(batches):
    sum = all_metrics[str(batch)][metric]


    vec = all_metrics[str(batch)][metric] / all_metrics[str(batch)]['gradients_L_norm']

    plt.plot(all_metrics[str(batch)]['iter'],
                vec,
                label="Batch Size " + str(batch) if batch != 5001 else "Batch Size 5000 L2",
                linewidth=8,
                color=jet(i))

plt.xlabel("Iterations")
plt.ylim(-0.1,1.1)
plt.grid()
plt.savefig(f'{metric}.pdf', bbox_inches='tight', format='pdf', dpi=300)

# metrics = ['cosine_Lhat_L', 'cosine_Lhat_Iinv_D', 'cosine_Lhat_alpha_D']

# fig = plt.figure(figsize=(16, 8))
# for metric in metrics:
#     for i, batch in enumerate([5000,5001]):
#         vec = all_metrics[str(batch)][metric]
#         plt.plot(all_metrics[str(batch)]['iter'],
#                  vec,
#                  label="Batch Size " + str(batch) if batch != 5001 else "Batch Size 5000 L2",
#                     linewidth=8,
#                     color=jet(i))

#     plt.xlabel('Iterations')
#     plt.grid()
#     #plt.ylim(-0.5,10)
#     #plt.xlim(0,50)
#     plt.savefig(f'{metric}.pdf', bbox_inches='tight', format='pdf', dpi=300)
#     plt.show()
#     plt.clf()



metrics = ['gradients_L_norm', 'gradients_Iinv_batch_norm', 'gradients_alpha_batch_norm']

for metric in metrics:

    for i, batch in enumerate(batches):

        sum = 0
        for metric2 in metrics:
            sum = sum + all_metrics[str(batch)][metric2]


        vec = all_metrics[str(batch)][metric] / sum
        vec = all_metrics[str(batch)][metric] / all_metrics[str(batch)]['gradients_Lbatch_norm']



        plt.plot(all_metrics[str(batch)]['iter'],
                 vec,
                 label="Batch Size " + str(batch) if batch != 5001 else "Batch Size 5000 L2",
                    linewidth=8,
                    color=jet(i))

    plt.xlabel('Iterations')
    plt.grid()
    plt.savefig(f'{metric}.pdf', bbox_inches='tight', format='pdf', dpi=300)
    plt.clf()


# projections = ['norm_projection_L_onto_Lhat', 'norm_projection_Iinv_D_onto_Lhat', 'norm_projection_alpha_D_onto_Lhat']

# for projection in projections:

#     for i, batch in enumerate([250, 5000]):
#         sum = all_metrics[str(batch)]['norm_projection_L_onto_Lhat']
#         sum  = sum + all_metrics[str(batch)]['norm_projection_Iinv_D_onto_Lhat']
#         sum  = sum  + all_metrics[str(batch)]['norm_projection_alpha_D_onto_Lhat']


#         if (metric == 'norm_projection_L_onto_Lhat'):
#             vec = all_metrics[str(batch)][projection] / sum
#         else:
#             vec = all_metrics[str(batch)][projection] / sum

#         plt.plot(all_metrics[str(batch)]['iter'],
#                  vec,
#                  label="Batch Size " + str(batch) if batch != 5001 else "Batch Size 5000 L2",
#                     linewidth=8,
#                     color=jet(i))

#     plt.title(projection)
#     plt.xlabel(projection)
#     plt.grid()
#     plt.legend()
#     #plt.ylim(-0.5,10)
#     #plt.xlim(0,50)
#     plt.show()

projections = ['norm_projection_L_onto_Lhat', 'norm_projection_Iinv_D_onto_Lhat', 'norm_projection_alpha_D_onto_Lhat']
labels = [r'$L(\bm{\theta})$', r'$\mathcal{I}^{-1}_{\bm{\theta}}(\alpha(\cdot, \bm{\theta}))$', r'$\alpha(\cdot, \bm{\theta})$']
for i, batch in enumerate([250, 5000]):
    #sum = np.sign(all_metrics[str(batch)]['cosine_Lhat_L'])*all_metrics[str(batch)]['norm_projection_L_onto_Lhat']
    #sum  = sum -np.sign(all_metrics[str(batch)]['cosine_Lhat_Iinv_D'])*all_metrics[str(batch)]['norm_projection_Iinv_D_onto_Lhat']
    #sum  = sum -np.sign(all_metrics[str(batch)]['cosine_Lhat_alpha_D'])*all_metrics[str(batch)]['norm_projection_alpha_D_onto_Lhat']

    #sum = all_metrics[str(batch)]['gradients_Lhat_norm']

    sum = all_metrics[str(batch)]['norm_projection_L_onto_Lhat']
    sum  = sum + all_metrics[str(batch)]['norm_projection_Iinv_D_onto_Lhat']
    sum  = sum + all_metrics[str(batch)]['norm_projection_alpha_D_onto_Lhat']
    fig = plt.figure(figsize=(16, 8))
    axis = plt.gca()
    for j, projection in enumerate(projections):
        if (projection == 'norm_projection_L_onto_Lhat'):
            vec = all_metrics[str(batch)][projection]/sum
        else:
            vec = all_metrics[str(batch)][projection]/sum
       
        axis.plot(all_metrics[str(batch)]['iter'],
                 vec,
                 label=labels[j],
                 color=jet(j),
                    linewidth=8)


    axis.grid()
    axis.set_xlabel('Iterations')
    axis.set_ylim(-0.01,1.01)
    plt.savefig(f'projections_{batch}.pdf', bbox_inches='tight', format='pdf', dpi=300)
    
    if i == 0:
        plt.show()
        plt.cla()
        lines, labels = axis.get_legend_handles_labels()
        legend = plt.legend(lines, labels, loc=0, ncol=4, fancybox=True, shadow=True)
        for line in legend.get_lines():
            line.set_linewidth(20)
        def export_legend(legend, filename="legend.pdf"):
            fig  = legend.figure
            fig.canvas.draw()
            bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(filename, dpi="figure", bbox_inches=bbox)


        export_legend(legend, "legend_norm_projection.pdf")
    
    plt.clf()




fig = plt.figure(figsize=(16, 8))

for i, batch in enumerate(batches):

    plt.scatter(all_metrics[str(batch)]['rate_B'],all_metrics[str(batch)]['gradients_Iinv_batch_norm'], 
                label=str(batch) if batch != 5001 else "Batch Size 5000 L2",
                s = 120,
                alpha = 0.9,
                color=jet(i))
    #plt.scatter(all_metrics[str(batch)]['rate_D'], all_metrics[str(batch)]['gradients_Iinv_D_norm'], label=str(batch))
#plt.ylim(-1.1, 1.1)
plt.ylabel(r'$\|\nabla_{\bm{\theta}} \mathcal{I}^{-1}_{\bm{\theta}}(\alpha(\cdot, \bm{\theta}))\|_2$')
plt.xlabel(r'$\alpha(\cdot, \bm{\theta})$')
plt.grid()
plt.savefig('rate_B_gradients_Iinv_batch_norm.pdf', bbox_inches='tight', format='pdf', dpi=300)
plt.clf()


# for batch in batches:
#     plt.plot(all_metrics[str(batch)]['iter'],all_metrics[str(batch)]['inv_B_approx'], label='sigma '+str(batch))
#     plt.plot(all_metrics[str(batch)]['iter'],all_metrics[str(batch)]['L'], label='L '+str(batch))
#     plt.plot(all_metrics[str(batch)]['iter'],all_metrics[str(batch)]['inv_B'], label='Iinv '+str(batch))
#     #plt.ylim(-1.1, 1.1)
#     plt.ylabel('')
#     plt.xlabel('iter')
#     plt.legend()
#     plt.show()

# for batch in batches:
#     plt.scatter(all_metrics[str(batch)]['inv_B'],np.minimum(all_metrics[str(batch)]['L'],all_metrics[str(batch)]['inv_B_approx']), label='Corr '+str(batch))
#     plt.scatter(all_metrics[str(batch)]['inv_B'],all_metrics[str(batch)]['inv_B'], label='Linear '+str(batch))
#     #plt.ylim(-1.1, 1.1)
#     plt.ylabel('inv_B_approx')
#     plt.xlabel('inv_B')
#     plt.legend()
#     plt.show()


# for batch in batches:
#     plt.scatter(all_metrics[str(batch)]['rate_B'],np.minimum(all_metrics[str(batch)]['L'],all_metrics[str(batch)]['inv_B_approx']), label='Corr '+str(batch))
#     plt.scatter(all_metrics[str(batch)]['rate_B'],all_metrics[str(batch)]['inv_B'], label='Linear '+str(batch))
#     #plt.ylim(-1.1, 1.1)
#     plt.ylabel('inv_B_approx')
#     plt.xlabel('rate_B')
#     plt.legend()
#     plt.show()



# for batch in batches:
#     print(np.corrcoef(all_metrics[str(batch)]['inv_B'],np.minimum(all_metrics[str(batch)]['L'],all_metrics[str(batch)]['inv_B_approx']))[0,1])



fig = plt.figure(figsize=(8, 5))
# plt.rcParams.update({
#     'font.size': 10,
#     'text.usetex': True,
#     'text.latex.preamble': r'\usepackage{amsfonts}\usepackage{bm}'
# })
# mpl.rc('font',family='Times New Roman')
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
# plt.rcParams['figure.figsize'] = (16, 9)
# fontsize = 46
# matplotlib.rcParams.update({'font.size': fontsize})

jet = plt.get_cmap('Dark2')
for i, batch in enumerate(batches):
    metrics = all_metrics[str(batch)]
    fig = plt.figure(figsize=(16, 8))
    axis = plt.gca()
    axis.scatter(metrics['iter'],
                 metrics['cosine_Lhat_Lbatch'],
                 label=r"$\nabla_{\bm{\theta}} \hat{L}(\cdot, \bm{\theta})$", s = 80, color=jet(0))
    axis.scatter(metrics['iter'],
                 metrics['cosine_Iinv'], 
                 label=r'$\nabla_{\bm{\theta}} {\mathcal{I}}^{-1}_{\bm{\theta}}(\alpha(\cdot, \bm{\theta}))$' , s = 80, color=jet(1))
    axis.scatter(metrics['iter'],
                 metrics['cosine_Alpha'], 
                 label=r'$\nabla_{\bm{\theta}} \alpha(\cdot, \bm{\theta})$', s = 80, color=jet(2))
    axis.set_ylim(-1.1, 1.1)

    axis.set_xlabel('Iterations')
    axis.grid()

    plt.savefig(f'cosine_{batch}.pdf', bbox_inches='tight', format='pdf', dpi=300)

    if i == len(batches) - 1:
        plt.show()
    plt.cla()


plt.cla()
lines, labels = axis.get_legend_handles_labels()
legend = plt.legend(lines, labels, loc=0, ncol=4, fancybox=True, shadow=True)

#change the marker size manually for both lines
legend.legend_handles[0]._sizes = [500]
legend.legend_handles[1]._sizes = [500]
legend.legend_handles[2]._sizes = [500]
def export_legend(legend, filename="legend.pdf"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

export_legend(legend, "legend_scatter.pdf")