# -*- coding: utf-8 -*-
"""
Created on Sun May 14 17:50:37 2023

@author: Mert
"""
from collections import defaultdict

import os
import pandas as pd

import seaborn as sns

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import os
import random
import numpy as np
import torch
from torch import nn

from utils import to_np

import matplotlib.patheffects as path_effects
import matplotlib.colors as cols

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils import create_array, find_csv_files_in_directory, dataset_in_csv

# #################### DATA PARAMETERS #############################

datasets = [
    'synthetic1', 'synthetic2', 'synthetic3', 'synthetic4', 'synthetic5'
    ]
increase_beta = False
beta = 1e-3
begining_point = 0
every = 1000
activation = 'elu'

increase_beta = 'inc_beta {}'.format(str(increase_beta))
beta = 'beta {}'.format(str(beta))

# #################### PLOTTING PARAMETERS #########################

plt.style.use("seaborn-bright")

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({
    # 'font.size': 8,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

SMALL_SIZE = 25
MEDIUM_SIZE = 40
BIGGER_SIZE = 40

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#################### PLOTTING PARAMETERS #########################

path = './epoch_results/'
csv_files = find_csv_files_in_directory(path)
csv_files = [x for x  in csv_files if activation in x]
csv_files = [x for x in csv_files if dataset_in_csv(x, datasets)]

betas = [float(csv_file.split('beta')[1].split(',')[0]) for csv_file in csv_files]
betas = list(set(betas))

plot_list = list()
for csv_file in csv_files:
    csv_file_split = csv_file.split('/')[-1].split('_')
    data = csv_file_split[0].split('\\')[-1]
    fold = csv_file_split[2]
    model = csv_file_split[3]
    net = csv_file_split[10].split('net ')[1].split(',')[0]
    beta = float(csv_file_split[10].split('beta')[-1].split(', ')[0])
    epoch_results = pd.read_csv(csv_file)
    for index, row in epoch_results.iterrows():
        plot_list.append(
            (
                model, net, beta, data, fold, index + 1, 
                row['Valid ELBO'], row['Valid RMSE'], row['Valid MSE PHI0'])
            )
to_zero = r'$\mathbb{E}_{\mathcal{D}}\left[\sum_{j\in\mathcal{S}} \mu_{\theta_j}(\mathbf{x})\right]$'
plot_list = pd.DataFrame(
    plot_list, columns = [
        'Model', 'Network', 'Beta', 'Data', 'Fold', 'Epoch',
        r'$\hat{\mathcal{V}}$', r'RMSE', to_zero]
    )

betas = sorted(betas)
col_size = len(datasets)

median = plot_list.groupby(
    ['Model', 'Network', 'Beta', 'Data', 'Epoch']
    ).mean([r'$\hat{\mathcal{V}}$', r'RMSE', to_zero]).reset_index()
std = plot_list.groupby(
    ['Model', 'Network', 'Beta', 'Data', 'Epoch']
    ).agg(np.std, ddof=0).reset_index()

fig, axes = plt.subplots(nrows=2, ncols=len(datasets), figsize=(25, 9))
nets = ['vanilla', 'masked']
for i, dataset in enumerate(datasets):
    for beta in betas:
        for net in nets:
            
            if net == 'masked':
                ax = axes[0][i]
            elif net == 'vanilla':
                ax = axes[1][i]
            
            median_selected = median[
                (median.Data == dataset) & (median.Beta == beta) & (median.Network == net)
                ][to_zero]
            std_selected = std[
                (std.Data == dataset) & (std.Beta == beta) & (std.Network == net)
                ][to_zero]

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
        
            ax.plot(
                x, y,  alpha=0.9, label=r'$\beta$' + ' {}'.format(beta),
                marker='^', markersize=5,
                )
            ax.fill_between(
                x, 
                y-2*err,
                y+2*err,
                alpha=0.1,
                # color=palette[i],
                )
            ax.axhline(0, color='red',linestyle='--')
            ax.legend(prop={'size': 20})
        if net == 'masked':
            ax.set_title(dataset, size=25)

            
    # if net == 'vanilla':
    #     ax[i].set_xlabel('Epoch', size=25)
    # if i == 0:
    #     ax.set_ylabel(to_zero, size=25)
    
    for label in plt.gca().get_yticklabels():
        if label.get_text() ==  '$\\mathdefault{0}$':  # Depending on the format it might be '0' or '0.0'
            label.set_color('red')

axes[0][0].set_ylabel('Masked network')        
axes[1][0].set_ylabel('Vanilla network')        
plt.tight_layout()
# fig.text(-0.02, 0.5, to_zero, va='center', rotation='vertical')
plt.savefig(
"./figures/diagnostics.pdf"
)