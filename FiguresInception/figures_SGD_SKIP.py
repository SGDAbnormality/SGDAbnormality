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


network = "INCEPTION_DISCARD"
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
skips = [0.0, 0.001, 0.005, 0.01]


for skip in skips:
    if skip != 0.0:
        name_model = f"{network}_{dataset_name}_{TRAIN_SIZE}_{TEST_SIZE}_250_{skip}"
    else:
        name_model = f"INCEPTION_{dataset_name}_{TRAIN_SIZE}_{TEST_SIZE}_250_0.0"


    with open(f'./metrics/metrics_{name_model}.pickle', 'rb') as f:
        metrics=pickle.load(f)

    for key in metrics.keys():
        metrics[key]=np.array(metrics[key])

        #if skip == 5001:
        #    metrics[key] = metrics[key][:82]
    all_metrics[str(skip)]=metrics

jet = plt.get_cmap('Set2')

for skip in skips:
    all_metrics[str(skip)]['gen']=all_metrics[str(skip)]['L']-all_metrics[str(skip)]['Lhat']

for skip in skips:
    all_metrics[str(skip)]['gen_skip']=all_metrics[str(skip)]['L']-all_metrics[str(skip)]['Lbatch']

for skip in skips:
    all_metrics[str(skip)]['iter']=all_metrics[str(skip)]['iter']


for metric in all_metrics[str(skips[0])].keys():
    all_metrics['0.001'][metric][0] = all_metrics['0.0'][metric][0]
    all_metrics['0.005'][metric][0] = all_metrics['0.0'][metric][0]
    all_metrics['0.01'][metric][0] = all_metrics['0.0'][metric][0]


fig = plt.figure(figsize=(16, 8))
metric = 'L'
axis = plt.gca()
for i, skip in enumerate(skips):
    plt.plot(all_metrics[str(skip)]['iter'],
             all_metrics[str(skip)][metric],
             label="Skip Size " + str(skip),
                linewidth=8,
                color=jet(i))
    plt.plot(all_metrics[str(skip)]['iter'],
             all_metrics[str(skip)]['Lhat'],
                linewidth=8,
                alpha = 0.5,
                color=jet(i))
plt.xlabel('Iterations')
plt.grid()
plt.savefig("L_Lhat_Skip.pdf", bbox_inches='tight', format='pdf', dpi=300)
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

export_legend(legend, "legend_skip.pdf")

plt.clf()


fig = plt.figure(figsize=(16, 8))
metric = 'variance'
for i, skip in enumerate(skips):
    plt.plot(all_metrics[str(skip)]['iter'], 
             all_metrics[str(skip)][metric], 
             label="skip Size " + str(skip),
                linewidth=8,
                color=jet(i))
plt.xlabel('Iterations')
plt.grid()
plt.savefig("variance_skip.pdf", bbox_inches='tight', format='pdf', dpi=300)
plt.clf()



fig = plt.figure(figsize=(16, 8))
metric = 'rate_D'
for i, skip in enumerate(skips):
    plt.plot(all_metrics[str(skip)]['iter'], 
             all_metrics[str(skip)][metric], 
             label="skip Size " + str(skip) if skip != 5001 else "skip Size 5000 L2",
                linewidth=8,
                color=jet(i))
    plt.plot(all_metrics[str(skip)]['iter'], 
             all_metrics[str(skip)]['rate_B'], 
             label="skip Size " + str(skip) if skip != 5001 else "skip Size 5000 L2",
                linewidth=8,
                alpha = 0.5,
                color=jet(i))
plt.xlabel('Iterations')
plt.grid()
plt.savefig("rate_D_skip.pdf", bbox_inches='tight', format='pdf', dpi=300)
plt.clf()

metrics = ['J_D', 'distance_Lhat_L', 'cosine_Lhat_L']

fig = plt.figure(figsize=(16, 8))
for metric in metrics:
    for i, batch in enumerate(skips):
        vec = all_metrics[str(batch)][metric]
        plt.plot(all_metrics[str(batch)]['iter'],
                 vec,
                 label="Batch Size " + str(batch) if batch != 5001 else "Batch Size 5000 L2",
                    linewidth=8,
                    color=jet(i))

    plt.xlabel("Iterations")
    plt.grid()
    plt.savefig(f'{metric}_skip.pdf', bbox_inches='tight', format='pdf', dpi=300)
    plt.clf()

metric = 'gradients_Iinv_D_norm'

for i, batch in enumerate(skips):
    sum = all_metrics[str(batch)][metric]


    vec = all_metrics[str(batch)][metric] / all_metrics[str(batch)]['gradients_L_norm']

    plt.plot(all_metrics[str(batch)]['iter'],
                vec,
                label="Batch Size " + str(batch) if batch != 5001 else "Batch Size 5000 L2",
                linewidth=8,
                color=jet(2+i) if i != 0 else jet(0))

plt.xlabel("Iterations")
plt.ylim(-0.1,1.1)
plt.grid()
plt.savefig(f'{metric}_skip.pdf', bbox_inches='tight', format='pdf', dpi=300)




fig = plt.figure(figsize=(16, 8))
axis = plt.gca()
labels = ["Batch Size 250", "Batch Size 500", "Batch Size 5000", "Batch Size 250 Skip 0.001", "Batch Size 250 Skip 0.005", "Batch Size 250 Skip 0.01"]
          
for i, label in enumerate(labels):
    batch = skips[0]
    sum = all_metrics[str(batch)][metric]


    vec = all_metrics[str(batch)][metric] / all_metrics[str(batch)]['gradients_L_norm']

    axis.plot(all_metrics[str(batch)]['iter'],
                vec,
                label=label,
                linewidth=8,
                color=jet(i) if i != 0 else jet(0))

axis.set_xlabel("Iterations")
axis.set_ylim(-0.1,1.1)
axis.grid()
plt.show()
lines, labels = axis.get_legend_handles_labels()
legend = plt.legend(lines, labels, loc=0, ncol=3, fancybox=True, shadow=True)
for line in legend.get_lines():
    line.set_linewidth(20)
def export_legend(legend, filename="legend.pdf"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

export_legend(legend, "legend_grad_Iinv.pdf")

plt.clf()
# fig = plt.figure(figsize=(16, 8))
# metric = 'rate_B'
# for i, skip in enumerate(skips):
#     plt.plot(all_metrics[str(skip)]['iter'], 
#              all_metrics[str(skip)][metric], 
#              label="skip Size " + str(skip) if skip != 5001 else "skip Size 5000 L2",
#                 linewidth=8,
#                 color=jet(i))
# plt.xlabel('Training Completion (\%)')
# plt.title(metric)
# plt.grid()
# plt.show()
# plt.clf()


# fig = plt.figure(figsize=(16, 8))
# metric = 'gen'
# for i, skip in enumerate(skips):
#     plt.plot(all_metrics[str(skip)]['iter'],
#              all_metrics[str(skip)][metric],
#              label="skip Size " + str(skip) if skip != 5001 else "skip Size 5000 L2",
#                 linewidth=8,
#                 color=jet(i))
# plt.xlabel('Training Completion (\%)')
# plt.title(metric)
# plt.legend()
# plt.grid()
# plt.show()
# plt.clf()

# fig = plt.figure(figsize=(16, 8))
# metric = 'gen_skip'
# for i, skip in enumerate(skips):
#     plt.plot(all_metrics[str(skip)]['iter'], 
#              all_metrics[str(skip)][metric], 
#              label="skip Size " + str(skip) if skip != 5001 else "skip Size 5000 L2",
#                 linewidth=8,
#                 color=jet(i))
# plt.xlabel('Training Completion (\%)')
# plt.grid()
# plt.legend()
# plt.show()
# plt.clf()


# fig = plt.figure(figsize=(16, 8))
# metric = 'lamb_rate_D'
# for i, skip in enumerate(skips):
#     plt.plot(all_metrics[str(skip)]['iter'],
#              all_metrics[str(skip)][metric],
#              label="skip Size " + str(skip) if skip != 5001 else "skip Size 5000 L2",
#                 linewidth=8,
#                 color=jet(i))
# plt.xlabel('Training Completion (\%)')
# plt.grid()
# plt.show()
# plt.clf()


# def cum_mean(arr):
#     cum_sum = np.cumsum(arr, axis=0)
#     for i in range(cum_sum.shape[0]):
#         if i == 0:
#             continue
#         print(cum_sum[i] / (i + 1))
#         cum_sum[i] =  cum_sum[i] / (i + 1)
#     return cum_sum

# N_ITERS=3000
# discards = {}
# for i, skip in enumerate(skips):
#     if i==0:
#         discards[str(skip)] = []
#     else:
#         with open(f"./models/INCEPTION_DISCARD_CIFAR10_0_0_250_{skip}/discarded_iterations.txt", 'r') as file:
#             discards[str(skip)] = np.loadtxt(file)

# for i, skip in enumerate(skips):
#     vec = np.zeros(N_ITERS)
#     for j, discard in enumerate(discards[str(skip)]):
#         vec[int(discard)]=1
#     plt.plot(np.arange(0,N_ITERS),
#              cum_mean(vec),
#              label="skip Size " + str(skip),
#                 linewidth=8,
#                 color=jet(i))

# plt.legend()
# plt.show()





# fig = plt.figure(figsize=(16, 8))
# for i, skip in enumerate(skips):
#     lamb = all_metrics[str(skip)]['lamb_rate_B']*np.sign(all_metrics[str(skip)]['gen_skip'])
#     J = all_metrics[str(skip)]['J_B']
#     plt.plot(all_metrics[str(skip)]['iter'],
#              np.exp(-J+all_metrics[str(skip)]['L']),
#              label="1",
#                 linewidth=8,
#                 color=jet(0))
#     plt.plot(all_metrics[str(skip)]['iter'],
#              lamb * np.exp(-J)/2,
#              label="2",
#                 linewidth=8,
#                 color=jet(1))
#     plt.plot(all_metrics[str(skip)]['iter'],
#              lamb**2 * np.exp(-J)/6,
#              label="3",
#                 linewidth=8,
#                 color=jet(2))
#     plt.plot(all_metrics[str(skip)]['iter'],
#              lamb**3 * np.exp(-J)/24,
#              label="4",
#                 linewidth=8,
#                 color=jet(3))
#     plt.title(str(skip))
#     plt.xlabel('Training Completion (\%)')
#     plt.grid()
#     plt.legend(loc='best')
#     plt.savefig('lamb_rate_D.pdf', bbox_inches='tight', format='pdf', dpi=300)
#     plt.show()
#     plt.clf()

# fig = plt.figure(figsize=(16, 8))
# for i, skip in enumerate(skips):
#     lamb = all_metrics[str(skip)]['lamb_rate_D']
#     J = all_metrics[str(skip)]['J_D']
#     factor = 1.0#all_metrics[str(skip)]['L'][-1]/np.cumsum(np.exp(-J))[-1]
#     plt.plot(all_metrics[str(skip)]['iter'],
#              np.cumsum(np.exp(-J))*factor,
#              label="1",
#                 linewidth=8,
#                 color=jet(0))
#     plt.plot(all_metrics[str(skip)]['iter'],
#              np.cumsum(lamb * np.exp(-J)/2)*factor,
#              label="2",
#                 linewidth=8,
#                 color=jet(1))
#     plt.plot(all_metrics[str(skip)]['iter'],
#              np.cumsum(lamb**2 * np.exp(-J)/6)*factor,
#              label="3",
#                 linewidth=8,
#                 color=jet(2))
#     plt.plot(all_metrics[str(skip)]['iter'],
#              np.cumsum(lamb**3 * np.exp(-J)/24)*factor,
#              label="4",
#                 linewidth=8,
#                 color=jet(3))
#     plt.title(str(skip))
#     plt.ylim(0,1.1)
#     plt.xlabel('Training Completion (\%)')
#     plt.grid()
#     plt.legend(loc='best')
#     plt.savefig('lamb_rate_D.pdf', bbox_inches='tight', format='pdf', dpi=300)
#     plt.show()
#     plt.clf()



# fig = plt.figure(figsize=(16, 8))
# for i, skip in enumerate(skips):
#     lamb = all_metrics[str(skip)]['lamb_rate_D']
#     J = all_metrics[str(skip)]['J_D']
#     factor = 1.0#all_metrics[str(skip)]['L'][-1]/np.cumsum(np.exp(-J))[-1]
#     plt.plot(all_metrics[str(skip)]['iter'],
#              np.cumsum(np.exp(-J))*factor,
#              label="1",
#                 linewidth=8,
#                 color=jet(0))
# plt.title(str(skip))
# plt.ylim(0,1.1)
# plt.xlabel('Training Completion (\%)')
# plt.grid()
# plt.legend(loc='best')
# plt.savefig('lamb_rate_D.pdf', bbox_inches='tight', format='pdf', dpi=300)
# plt.show()
# plt.clf()

# fig = plt.figure(figsize=(16, 8))
# for i, skip in enumerate(skips):
#     lamb = all_metrics[str(skip)]['lamb_rate_B']
#     J = all_metrics[str(skip)]['J_B']
#     plt.plot(all_metrics[str(skip)]['iter'],
#              lamb * np.exp(-J)/2,
#              label=skip,
#                 linewidth=8,
#                 color=jet(i))
# plt.title("Var")
# plt.xlabel('Training Completion (\%)')
# plt.grid()
# plt.legend(loc='best')
# plt.savefig('lamb_rate_D.pdf', bbox_inches='tight', format='pdf', dpi=300)
# plt.show()
# plt.clf()


# fig = plt.figure(figsize=(16, 8))
# metric = 'lamb_rate_B'
# for i, skip in enumerate(skips):
#     plt.plot(all_metrics[str(skip)]['iter'],
#              all_metrics[str(skip)][metric],
#              label="skip Size " + str(skip) if skip != 5001 else "skip Size 5000 L2",
#                 linewidth=8,
#                 color=jet(i))
# plt.xlabel('Training Completion (\%)')
# plt.grid()
# plt.ylim(-2,20)
# plt.savefig('lamb_rate_D.pdf', bbox_inches='tight', format='pdf', dpi=300)
# plt.show()
# plt.clf()

# fig = plt.figure(figsize=(16, 8))
# metric = 'gen_skip'
# for i, skip in enumerate(skips):
#     plt.scatter(all_metrics[str(skip)][metric],
#                 all_metrics[str(skip)]['gradients_alpha_skip_norm'],
#              label="skip Size " + str(skip) if skip != 5001 else "skip Size 5000 L2",
#                 linewidth=8,
#                 color=jet(i))
# plt.xlabel('Training Completion (\%)')
# plt.grid()
# plt.savefig('gen_skip.pdf', bbox_inches='tight', format='pdf', dpi=300)
# plt.show()
# plt.clf()





# metrics = ['cosine_Lskip_Iinv_D', 'cosine_Lskip_alpha_D']

# fig = plt.figure(figsize=(16, 8))
# for metric in metrics:
#     for i, skip in enumerate([250,5000]):
#         vec = all_metrics[str(skip)][metric]
#         plt.plot(all_metrics[str(skip)]['iter'],
#                  vec,
#                  label="skip Size " + str(skip) if skip != 5001 else "skip Size 5000 L2",
#                     linewidth=8,
#                     color=jet(i))

#     plt.xlabel("Training Completion (\%)")
#     plt.grid()
#     #plt.ylim(-0.5,10)
#     #plt.xlim(0,50)
#     plt.savefig(f'{metric}.pdf', bbox_inches='tight', format='pdf', dpi=300)
#     plt.show()
#     plt.clf()



# metrics = ['cosine_Lhat_L', 'cosine_Lhat_Iinv_D', 'cosine_Lhat_alpha_D']

# fig = plt.figure(figsize=(16, 8))
# for metric in metrics:
#     for i, skip in enumerate([5000,5001]):
#         vec = all_metrics[str(skip)][metric]
#         plt.plot(all_metrics[str(skip)]['iter'],
#                  vec,
#                  label="skip Size " + str(skip) if skip != 5001 else "skip Size 5000 L2",
#                     linewidth=8,
#                     color=jet(i))

#     plt.xlabel("Training Completion (\%)")
#     plt.grid()
#     #plt.ylim(-0.5,10)
#     #plt.xlim(0,50)
#     plt.savefig(f'{metric}.pdf', bbox_inches='tight', format='pdf', dpi=300)
#     plt.show()
#     plt.clf()



# metrics = ['gradients_L_norm', 'gradients_Iinv_skip_norm', 'gradients_alpha_skip_norm']

# for metric in metrics:

#     for i, skip in enumerate(skips):

#         sum = 0
#         for metric2 in metrics:
#             sum = sum + all_metrics[str(skip)][metric2]


#         vec = all_metrics[str(skip)][metric] / sum
#         vec = all_metrics[str(skip)][metric] / all_metrics[str(skip)]['gradients_Lskip_norm']



#         plt.plot(all_metrics[str(skip)]['iter'],
#                  vec,
#                  label="skip Size " + str(skip) if skip != 5001 else "skip Size 5000 L2",
#                     linewidth=8,
#                     color=jet(i))

#     plt.xlabel("Training Completion (\%)")
#     plt.grid()
#     plt.savefig(f'{metric}.pdf', bbox_inches='tight', format='pdf', dpi=300)
#     plt.clf()


# # projections = ['norm_projection_L_onto_Lhat', 'norm_projection_Iinv_D_onto_Lhat', 'norm_projection_alpha_D_onto_Lhat']

# # for projection in projections:

# #     for i, skip in enumerate([250, 5000]):
# #         sum = all_metrics[str(skip)]['norm_projection_L_onto_Lhat']
# #         sum  = sum + all_metrics[str(skip)]['norm_projection_Iinv_D_onto_Lhat']
# #         sum  = sum  + all_metrics[str(skip)]['norm_projection_alpha_D_onto_Lhat']


# #         if (metric == 'norm_projection_L_onto_Lhat'):
# #             vec = all_metrics[str(skip)][projection] / sum
# #         else:
# #             vec = all_metrics[str(skip)][projection] / sum

# #         plt.plot(all_metrics[str(skip)]['iter'],
# #                  vec,
# #                  label="skip Size " + str(skip) if skip != 5001 else "skip Size 5000 L2",
# #                     linewidth=8,
# #                     color=jet(i))

# #     plt.title(projection)
# #     plt.xlabel(projection)
# #     plt.grid()
# #     plt.legend()
# #     #plt.ylim(-0.5,10)
# #     #plt.xlim(0,50)
# #     plt.show()

# projections = ['norm_projection_L_onto_Lhat', 'norm_projection_Iinv_D_onto_Lhat', 'norm_projection_alpha_D_onto_Lhat']
# labels = [r'$L(\bm{\theta})$', r'$\mathcal{I}^{-1}_{\bm{\theta}}(\alpha(\cdot, \bm{\theta}))$', r'$\alpha(\cdot, \bm{\theta})$']
# for i, skip in enumerate([250, 5000]):
#     #sum = np.sign(all_metrics[str(skip)]['cosine_Lhat_L'])*all_metrics[str(skip)]['norm_projection_L_onto_Lhat']
#     #sum  = sum -np.sign(all_metrics[str(skip)]['cosine_Lhat_Iinv_D'])*all_metrics[str(skip)]['norm_projection_Iinv_D_onto_Lhat']
#     #sum  = sum -np.sign(all_metrics[str(skip)]['cosine_Lhat_alpha_D'])*all_metrics[str(skip)]['norm_projection_alpha_D_onto_Lhat']

#     #sum = all_metrics[str(skip)]['gradients_Lhat_norm']

#     sum = all_metrics[str(skip)]['norm_projection_L_onto_Lhat']
#     sum  = sum + all_metrics[str(skip)]['norm_projection_Iinv_D_onto_Lhat']
#     sum  = sum + all_metrics[str(skip)]['norm_projection_alpha_D_onto_Lhat']
#     fig = plt.figure(figsize=(16, 8))
#     axis = plt.gca()
#     for j, projection in enumerate(projections):
#         if (projection == 'norm_projection_L_onto_Lhat'):
#             vec = all_metrics[str(skip)][projection]/sum
#         else:
#             vec = all_metrics[str(skip)][projection]/sum
       
#         axis.plot(all_metrics[str(skip)]['iter'],
#                  vec,
#                  label=labels[j],
#                  color=jet(j),
#                     linewidth=8)


#     axis.grid()
#     axis.set_xlabel('Training Completion (\%)')
#     axis.set_ylim(-0.01,1.01)
#     plt.savefig(f'projections_{skip}.pdf', bbox_inches='tight', format='pdf', dpi=300)
    
#     if i == 0:
#         plt.show()
#         plt.cla()
#         lines, labels = axis.get_legend_handles_labels()
#         legend = plt.legend(lines, labels, loc=0, ncol=4, fancybox=True, shadow=True)
#         for line in legend.get_lines():
#             line.set_linewidth(20)
#         def export_legend(legend, filename="legend.pdf"):
#             fig  = legend.figure
#             fig.canvas.draw()
#             bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#             fig.savefig(filename, dpi="figure", bbox_inches=bbox)


#         export_legend(legend, "legend_norm_projection.pdf")
    
#     plt.clf()




# fig = plt.figure(figsize=(16, 8))

# for i, skip in enumerate(skips):

#     plt.scatter(all_metrics[str(skip)]['rate_B'],all_metrics[str(skip)]['gradients_Iinv_skip_norm'], 
#                 label=str(skip) if skip != 5001 else "skip Size 5000 L2",
#                 s = 120,
#                 alpha = 0.9,
#                 color=jet(i))
#     #plt.scatter(all_metrics[str(skip)]['rate_D'], all_metrics[str(skip)]['gradients_Iinv_D_norm'], label=str(skip))
# #plt.ylim(-1.1, 1.1)
# plt.ylabel(r'$\|\nabla_{\bm{\theta}} \mathcal{I}^{-1}_{\bm{\theta}}(\alpha(\cdot, \bm{\theta}))\|_2$')
# plt.xlabel(r'$\alpha(\cdot, \bm{\theta})$')
# plt.grid()
# plt.savefig('rate_B_gradients_Iinv_skip_norm.pdf', bbox_inches='tight', format='pdf', dpi=300)
# plt.clf()


# # for skip in skips:
# #     plt.plot(all_metrics[str(skip)]['iter'],all_metrics[str(skip)]['inv_B_approx'], label='sigma '+str(skip))
# #     plt.plot(all_metrics[str(skip)]['iter'],all_metrics[str(skip)]['L'], label='L '+str(skip))
# #     plt.plot(all_metrics[str(skip)]['iter'],all_metrics[str(skip)]['inv_B'], label='Iinv '+str(skip))
# #     #plt.ylim(-1.1, 1.1)
# #     plt.ylabel('')
# #     plt.xlabel('iter')
# #     plt.legend()
# #     plt.show()

# # for skip in skips:
# #     plt.scatter(all_metrics[str(skip)]['inv_B'],np.minimum(all_metrics[str(skip)]['L'],all_metrics[str(skip)]['inv_B_approx']), label='Corr '+str(skip))
# #     plt.scatter(all_metrics[str(skip)]['inv_B'],all_metrics[str(skip)]['inv_B'], label='Linear '+str(skip))
# #     #plt.ylim(-1.1, 1.1)
# #     plt.ylabel('inv_B_approx')
# #     plt.xlabel('inv_B')
# #     plt.legend()
# #     plt.show()


# # for skip in skips:
# #     plt.scatter(all_metrics[str(skip)]['rate_B'],np.minimum(all_metrics[str(skip)]['L'],all_metrics[str(skip)]['inv_B_approx']), label='Corr '+str(skip))
# #     plt.scatter(all_metrics[str(skip)]['rate_B'],all_metrics[str(skip)]['inv_B'], label='Linear '+str(skip))
# #     #plt.ylim(-1.1, 1.1)
# #     plt.ylabel('inv_B_approx')
# #     plt.xlabel('rate_B')
# #     plt.legend()
# #     plt.show()



# # for skip in skips:
# #     print(np.corrcoef(all_metrics[str(skip)]['inv_B'],np.minimum(all_metrics[str(skip)]['L'],all_metrics[str(skip)]['inv_B_approx']))[0,1])



# fig = plt.figure(figsize=(8, 5))
# # plt.rcParams.update({
# #     'font.size': 10,
# #     'text.usetex': True,
# #     'text.latex.preamble': r'\usepackage{amsfonts}\usepackage{bm}'
# # })
# # mpl.rc('font',family='Times New Roman')
# # matplotlib.rcParams['pdf.fonttype'] = 42
# # matplotlib.rcParams['ps.fonttype'] = 42
# # plt.rcParams['figure.figsize'] = (16, 9)
# # fontsize = 46
# # matplotlib.rcParams.update({'font.size': fontsize})

# jet = plt.get_cmap('Dark2')
# for i, skip in enumerate(skips):
#     metrics = all_metrics[str(skip)]
#     fig = plt.figure(figsize=(10, 7))
#     axis = plt.gca()
#     axis.scatter(metrics['iter'],metrics['cosine_Iinv'], label=r"$\nabla_{\bm{\theta}} \hat{L}(\cdot, \bm{\theta})$", s = 80, color=jet(0))
#     axis.scatter(metrics['iter'],metrics['cosine_Alpha'], label=r'$\nabla_{\bm{\theta}} {\mathcal{I}}^{-1}_{\bm{\theta}}(\alpha(\cdot, \bm{\theta}))$' , s = 80, color=jet(1))
#     axis.scatter(metrics['iter'],metrics['cosine_Lhat_Lskip'], label=r'$\nabla_{\bm{\theta}} \alpha(\cdot, \bm{\theta})$', s = 80, color=jet(2))
#     axis.set_ylim(-1.1, 1.1)

#     axis.set_xlabel('Training Completion (\%)')
#     axis.grid()

#     plt.savefig(f'cosine_{skip}.pdf', bbox_inches='tight', format='pdf', dpi=300)

#     if i == len(skips) - 1:
#         plt.show()
#     plt.cla()


# plt.cla()
# lines, labels = axis.get_legend_handles_labels()
# legend = plt.legend(lines, labels, loc=0, ncol=4, fancybox=True, shadow=True)

# #change the marker size manually for both lines
# legend.legend_handles[0]._sizes = [500]
# legend.legend_handles[1]._sizes = [500]
# legend.legend_handles[2]._sizes = [500]
# def export_legend(legend, filename="legend.pdf"):
#     fig  = legend.figure
#     fig.canvas.draw()
#     bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#     fig.savefig(filename, dpi="figure", bbox_inches=bbox)

# export_legend(legend, "legend_scatter.pdf")