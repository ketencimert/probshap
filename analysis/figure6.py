# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 18:09:53 2023

@author: Mert
"""

from collections import defaultdict

import os
import pandas as pd

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

import numpy as np
import random

import torch
import os

from datasets import load_dataset

from utils import prepare_fold
from scipy.stats import gaussian_kde

from itertools import combinations
from scipy.stats import wasserstein_distance

from sklearn.neighbors import KernelDensity
from scipy.stats import entropy

from utils import to_np

import matplotlib.pyplot as plt

from itertools import chain, combinations
from math import factorial

import matplotlib

from scipy.spatial.distance import cdist


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

def energy_distance(X, Y):
    """
    Computes the two-sample energy distance between samples X and Y.

    Parameters:
    - X: First sample, a matrix of size [n_samples_X, n_features].
    - Y: Second sample, a matrix of size [n_samples_Y, n_features].

    Returns:
    - Energy distance between X and Y.
    """
    XX = cdist(X, X)
    YY = cdist(Y, Y)
    XY = cdist(X, Y)

    # Expectations
    avg_XX = np.mean(XX)
    avg_YY = np.mean(YY)
    avg_XY = np.mean(XY)

    return 2 * avg_XY - avg_XX - avg_YY

def shapley_weights(lst):
    n = len(lst)
    # Get all combinations of the list
    all_combinations = list(chain.from_iterable(combinations(lst, r) for r in range(n+1)))

    weights = []
    for combo in all_combinations:
        s_size = len(combo)
        if 0<s_size<n:
            p_s = (n - s_size) * factorial(s_size) * factorial(n - s_size - 1) / factorial(n)
            p_s += s_size * factorial(s_size - 1) * factorial(n - s_size) / factorial(n)
        elif s_size == 0:
            p_s = (n - s_size) * factorial(s_size) * factorial(n - s_size - 1) / factorial(n)
        elif s_size == n:
            p_s = s_size * factorial(s_size - 1) * factorial(n - s_size) / factorial(n)
        weights.append(p_s)
    all_combinations = [list(x) for x in all_combinations]
    weights = np.asarray(weights) / sum(weights)
    return all_combinations, weights

def compute_kl_divergence(sample_p, sample_q, bins=100):
    """
    Compute KL divergence between two samples using histogram estimation.
    
    Parameters:
    - sample_p, sample_q: Arrays of samples from the two distributions.
    - bins: Number of bins for histogram estimation. Can be an int or sequence.
    
    Returns:
    - KL divergence value.
    """
    
    # Calculate histograms
    hist_p, _ = np.histogram(sample_p, bins=bins, density=True)
    hist_q, _ = np.histogram(sample_q, bins=bins, density=True)
    
    # Replace zeros with a small value to avoid log(0) issue in KL computation
    hist_p = hist_p + 1e-10
    hist_q = hist_q + 1e-10
    
    # Compute KL divergence
    return entropy(hist_p, hist_q)


def find_pth_files_in_directory(path):
    pth_files = []

    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.pth'):
                full_path = os.path.join(dirpath, filename)
                pth_files.append(full_path)

    return pth_files 

##############################################################################


activation = 'elu'
datasets = [
    'synthetic1', 'synthetic2', 'synthetic3', 'synthetic4', 'synthetic5',
    ]
beta = 1

##############################################################################

SEED = 11
random.seed(SEED), np.random.seed(SEED), torch.manual_seed(SEED)
kld_results = []

for dataset in datasets:

    path = './model_checkpoints'
    pth_files = find_pth_files_in_directory(path)
    pth_files = [
        p for p in pth_files if (dataset in p) & (activation in p) & (
            'beta {}'.format(beta) in p
            )
        ]
    
    #we will plot the joint f1, f2 after marginalizing x1 and x2 for example
    
    
    with torch.no_grad():
        for pth_file in pth_files:
            
            # Set the general seed to default that the models are trained on:
            SEED = 11
            random.seed(SEED), np.random.seed(SEED), torch.manual_seed(SEED)
            # Load the model, it can be masked or vanilla:
            model = torch.load(pth_file, map_location='cuda').eval()
            #Get model details to log the results
            pth_file_split = pth_file.split('/')[-1].split('_')
            fold = int(pth_file_split[3])
            net = pth_file_split[-3].split('net')[1].split(',')[0][1:]
            #Load the data, this is why random seed is important
            data, features, dtypes, target_scale, \
            predictive_distribution = load_dataset(
                dataset=dataset,
                )
            x, y = data[0]
            d_in = x.shape[1]
            n = len(x)
            tr_size = int(n * 0.7)
            #We will do this for different folds:
            folds = np.array(list(range(5)) * n)[:n]
            np.random.shuffle(folds)        
            #load the train test valiadtion we used during training:
            tr_dataloader, val_dataloader, te_dataloader, _ = prepare_fold(
                x, y, 1024, fold, folds,
               'cuda', torch.float32, dtypes, True
                )
            #Get the train test valid splits:
            x_tr, y_tr = [
                torch.tensor(x).cuda() for x in tr_dataloader.dataset.__getds__()
                ]
            x_val, y_val = [
                torch.tensor(x).cuda() for x in val_dataloader.dataset.__getds__()
                ]
            x_te, y_te = [
                torch.tensor(x).cuda() for x in val_dataloader.dataset.__getds__()
                ]
            #Concatenate the data:
            x = torch.cat([x_tr, x_val, x_te])
            y = torch.cat([y_tr, y_val, y_te])
            #Now, we generate the marginals and their usage frequency when computing Shapley values:
            marginals, probs = shapley_weights(list(range(x.shape[-1])))
            #We first iterate over marginals:
            for marginal, prob in zip(marginals, probs):
                #For each marginal the initial dist metrics are 0
                mmd_value = 0
                kl_value = 0
                #We wll iterate over 5 different seeds
                seeds = [110, 12, 130, 150, 190]
                #We will sample grids to compute the kldiv for histogram approximation
                grids = np.random.randint(10,200, 20)
                #For each marginal and grid
                for i, grid in enumerate(grids):
                    #For each seed that we will use within that grid
                    for seed in seeds:
                        #We set the new seed to generate various samples from the model
                        random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
                        #We identify masks
                        missing = torch.zeros_like(x)
                        missing[:,marginal] = 1
                        #We build the ground truth
                        sample_p = to_np(
                            torch.cat([
                            x[:,marginal].view(x.size(0), -1), y.view(x.size(0), -1)],
                            -1
                            )
                            )
                        #We generate from the model, it can be masked model or vanilla model
                        sample_q = to_np(
                            torch.cat([
                            x[:,marginal].view(x.size(0), -1), model.sample(
                            x,
                            missing, False
                            ).view(x.size(0), -1)],
                            -1
                            )
                            )
                        try:
                            sample_p = sample_p.squeeze(-1)
                            sample_q = sample_q.squeeze(-1)
                        except:
                            pass
                        #We calculate the jensen-shannon div
                        kl_value += prob * 1/2 * (
                            compute_kl_divergence(
                                sample_p, sample_q, grid
                                ) + compute_kl_divergence(sample_q, sample_p, grid)
                            ) / len(seeds)
                        #No we don't want to calculate energy based second metric for every grid. so we only do this if i=0
                        #we still iterate over the seeds when i=0 hence we average
                        if i == 0:
                            M=ot.dist(
                                sample_p.reshape(sample_p.shape[0], -1),
                                sample_q.reshape(sample_p.shape[0], -1), 
                                metric='euclidean'
                                )
                            mmd_value += prob * ot.emd2(
                                np.ones(sample_p.shape[0]) / sample_p.shape[0],
                                np.ones(sample_p.shape[0]) / sample_p.shape[0],
                                M, 
                                numItermax=5
                                ) / len(seeds)
                    #when the computation is over, append:
                    kld_results.append(
                        (dataset,
                         net, 
                         grid, 
                         fold, 
                         tuple(marginal),
                         'js',
                         kl_value, 
                         pth_file)
                        )
                    kld_results.append(
                        (dataset,
                         net, 
                         grid, 
                         fold, 
                         tuple(marginal),
                         'mmd',
                         mmd_value, 
                         pth_file)
                        )


kld_results = pd.DataFrame(kld_results, columns = [
    'Data', 'Network', 'Grid Size', 'Fold', 'Marginal', 'Metric', 'Value', 'PTH']
    )

plt.figure(figsize=(5, 3))
plt.style.use("seaborn-bright")

kld = kld_results[kld_results.Metric == 'js']
mmd =  kld_results[kld_results.Metric == 'mmd']
df_kld = kld.groupby(['Data', 'Network', 'Grid Size', 'Fold']).sum().reset_index(drop = False)
df_kld = df_kld.groupby(['Data', 'Network', 'Fold']).mean().reset_index(drop = False)

df_mmd = mmd.groupby(['Data', 'Network', 'Fold']).sum().reset_index(drop = False)
df_mmd = df_mmd.groupby(['Data', 'Network', 'Fold']).mean().reset_index(drop = False)


# mean = df.groupby(['Data', 'Network']).mean()
# std = df.groupby(['Data', 'Network']).std() / 20 ** 0.5


fig, ax = plt.subplots(nrows=2, ncols=len(datasets), figsize=(18, 7))

for i, dataset in enumerate(datasets):

    values = np.concatenate([
    df_kld[df_kld.Data == dataset][df_kld.Network == 'masked']['Value'].values.reshape(-1,1),
    df_kld[df_kld.Data == dataset][df_kld.Network == 'vanilla']['Value'].values.reshape(-1,1)
    ], -1
        )
    
    # rectangular box plot
    bplot1 = ax[0][i].boxplot(values,
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=[r'masked', r'vanilla'],
                         showmeans=True
                         )  # will be used to label x-ticks
    ax[0][i].set_title(r'{}'.format(dataset))
    if i == 0:
        ax[0][i].set_ylabel(r'JS-Divergence')
    # fill with colors
    colors = ['pink', 'lightblue', 'lightgreen']
    for bplot in (bplot1, bplot1):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    values = np.concatenate([
    df_mmd[df_mmd.Data == dataset][df_mmd.Network == 'masked']['Value'].values.reshape(-1,1),
    df_mmd[df_mmd.Data == dataset][df_mmd.Network == 'vanilla']['Value'].values.reshape(-1,1)
    ], -1
        )
    
    # rectangular box plot
    bplot1 = ax[1][i].boxplot(values,
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=[r'masked', r'vanilla'],
                         showmeans=True
                         )  # will be used to label x-ticks
    ax[1][i].set_title(r'{}'.format(dataset))
    if i == 0:
        ax[1][i].set_ylabel(r'W-Distance')
    # fill with colors
    colors = ['pink', 'lightblue', 'lightgreen']
    for bplot in (bplot1, bplot1):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)  

    
plt.tight_layout()
plt.savefig(
"./figures/networks.pdf"
)
#load the masked model
# masked_network = torch.load()


#you want to simulate and compare ys also you might want to show fs?