# # -*- coding: utf-8 -*-
# """
# Created on Sun May 14 17:50:37 2023

# @author: Mert
# """
# from collections import defaultdict

# import os
# import pandas as pd

# import seaborn as sns

# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt

# import os
# import random
# import numpy as np
# import torch
# from torch import nn

# from utils import to_np

# import matplotlib.patheffects as path_effects
# import matplotlib.colors as cols

# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

# def create_array(n, k):
#     return [(i % k == 0) for i in range(n)]

# #################### DATA PARAMETERS #############################

# datasets = [
#     'synthetic1', 'synthetic2', 'synthetic3', 
#     # 'medical', 'bike','parkinsons'
#     ]

architecture = 'masked'
increase_beta = False
beta = 1e-2
begining_point = 0
every = 100

# increase_beta = 'inc_beta {}'.format(str(increase_beta))
# beta = 'beta {}'.format(str(beta))

# #################### PLOTTING PARAMETERS #########################

# plt.style.use('seaborn-paper')
# sns.set_palette("Dark2")
# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'
# plt.rcParams.update({
#     # 'font.size': 8,
#     'text.usetex': True,
#     'text.latex.preamble': r'\usepackage{amsfonts}'
# })

# SMALL_SIZE = 20
# MEDIUM_SIZE = 40
# BIGGER_SIZE = 40

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# #################### PLOTTING PARAMETERS #########################

# def is_in(x, y):
#     value = False
#     for e in x:
#         if e in y:
#             value = True
#     return value

# path = './epoch_results/'
# all_files = os.listdir(path)
# csv_files = []
# for epoch_results in all_files:
#     if is_in(datasets, epoch_results):
#         csv_files.append(
#             path + '{}'.format(epoch_results)
#             )

# csv_files = [x for x  in csv_files if increase_beta in x]
# csv_files = [x for x  in csv_files if architecture in x]

# betas = [float(csv_file.split('beta')[1].split(',')[0]) for csv_file in csv_files]
# betas = list(set(betas))

# plot_list = list()
# for csv_file in csv_files:
#     csv_file_split = csv_file.split('/')[-1].split('_')
#     data = csv_file_split[0]
#     fold = csv_file_split[2]
#     model = csv_file_split[3]
#     beta = float(csv_file.split('beta')[1].split(',')[0])
#     epoch_results = pd.read_csv(csv_file)
#     for index, row in epoch_results.iterrows():
#         plot_list.append(
#             (
#                 model, beta, data, fold, index + 1, 
#                 row['Valid ELBO'], row['Valid RMSE'], row['Valid MSE PHI0'])
#             )
# to_zero = r'$\mathbb{E}_{\mathcal{D}}[\sum_{j\in\mathcal{S}} \mu_{\theta_j}(\mathbf{x})]$'
# plot_list = pd.DataFrame(
#     plot_list, columns = [
#         'Model', 'Beta', 'Data', 'Fold', 'Epoch',
#         r'$\hat{\mathcal{V}}$', r'RMSE', to_zero]
#     )
betas = sorted(betas)

palette = sns.color_palette('dark')

col_size = len(plot_list.keys()) - 4

median = plot_list.groupby(
    ['Model','Beta', 'Data', 'Epoch']
    ).mean([r'$\hat{\mathcal{V}}$', r'RMSE', to_zero]).reset_index()
std = plot_list.groupby(
    ['Model', 'Beta', 'Data', 'Epoch']
    ).agg(np.std, ddof=0).reset_index()

fig, ax = plt.subplots(nrows=1, ncols=len(datasets), figsize=(15, 4))

for i, dataset in enumerate(datasets):
    for beta in betas:
        median_selected = median[(median.Data == dataset) & (median.Beta == beta)][to_zero]
        std_selected = std[(std.Data == dataset) & (std.Beta == beta)][to_zero]
    
        indicator = list(std_selected.values == 0)
    
        try:
            cut_off = int(indicator.index(True) * 0.8)
        except:
            cut_off = len(indicator)
    
        median_selected = median_selected[begining_point:cut_off]
        std_selected = std_selected[begining_point:cut_off]
        x = range(len(median_selected))
    
        fltr = create_array(cut_off - begining_point, every)
    
        median_selected = median_selected[fltr]
        std_selected = std_selected[fltr]
        x = np.asarray(x)[fltr]
        
        y = median_selected.values
        err = std_selected.values
        
        ax[i].plot(
            x, y,  alpha=0.9, label=r'$\beta$' + ' {}'.format(beta),
            marker='^', markersize=5,
            )
        ax[i].fill_between(
            x, 
            y-2*err,
            y+2*err,
            alpha=0.1,
            # color=palette[i],
            )
        ax[i].set_title(dataset, size=25)
        ax[i].axhline(0, color='red',linestyle='--')
    ax[i].legend(prop={'size': 20})
    ax[i].set_xlabel('Epoch', size=25)
    if i == 0:
        ax[i].set_ylabel(to_zero, size=25)
    
    for label in plt.gca().get_yticklabels():
        if label.get_text() ==  '$\\mathdefault{0}$':  # Depending on the format it might be '0' or '0.0'
            label.set_color('red')

# plt.title(r'$\beta=$' + ' {}'.format(beta.split(' ')[-1]))
plt.legend()
plt.tight_layout()
plt.savefig(
"./figures/diagnostics.pdf".format(beta, increase_beta)
)


