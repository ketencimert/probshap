# -*- coding: utf-8 -*-
'''
Created on Tue Oct  4 18:50:25 2022

@author: Mert
'''

import argparse
import os
import random

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import torch

from datasets import load_dataset

from utils import (
    transform, prepare_fold, to_np, get_shapley_values, load_model, train_fastshap
    )

import shap
from tqdm import tqdm
from collections import defaultdict

import time
import pickle

import pandas as pd


l2 = dict()
time_memory = dict()
SEED = 11
random.seed(SEED), np.random.seed(SEED), torch.manual_seed(SEED)

data_names = ['icu']
model_names = (
    ('VariationalShapley', 'FEEDFORWARD'), 
    # ('VariationalShapley', 'KERNEL_8'),
    ('VariationalShapley', 'KERNEL_16'),
    ('VariationalShapley', 'KERNEL_32'),
    ('VariationalShapley', 'KERNEL_64'),    # ('XGB', 'KERNEL'), 
    ('VariationalShapley', 'KERNEL_128'),    # ('DNN', 'KERNEL')
    ('VariationalShapley', 'KERNEL_256'),    # ('EBM', 'KERNEL'),
    ('VariationalShapley', 'KERNEL_512'),
    ('VariationalShapley', 'KERNEL_1024'),
    )
fold = 'X'
data_sample_size = 20
device = 'cuda'
act = 'elu'

shapley_values = defaultdict(list)

for model_name in model_names:
    
    for data_name in data_names:
        
        tensor_dtype = torch.float
                
        data, features, dtypes, target_scale, \
            predictive_distribution  = load_dataset(
            dataset=data_name,
            )
        _, _, _, stats = prepare_fold(
            *data[0], 1024, 1, np.asarray([0]),
            device, tensor_dtype, dtypes, args.preprocess
            )

        x, y, fs, ys, sigmas = data[1]
        x = transform(x, stats)
        x = torch.tensor(x, dtype=torch.float).to(device)
        x_sample = shap.sample(x, data_sample_size)
        
        model = load_model(
            model_name=model_name[0], 
            prior='masked', 
            act=act,
            beta=1, 
            inc_beta='False', 
            fold=fold, 
            dataset=data_name, 
            preprocess=True,
            device='cuda'
            )
        if 'KERNEL' in model_name[1]:
            nsamples = int(model_name[1].split('_')[-1])
            estimate = model_name[1].split('_')[0]
        else:
            estimate = model_name[1]
        shapley_values[model_name] = get_shapley_values(
            model=model, 
            estimate=estimate,
            features=features,
            dtypes=dtypes,
            x=x_sample,
            stats=stats,
            nsamples=nsamples,
            )

# for key in shapley_values.keys():
#     shapley_values[key] = shapley_values[key] - np.mean(shapley_values[key], 0)
covariance_matrix = defaultdict(dict)
for key1 in shapley_values.keys():
    for key2 in shapley_values.keys():
        # a = shapley_values[key1] - shapley_values[key1].mean(0)
        # b = shapley_values[key2] - shapley_values[key2].mean(0)
        # a = a / (a.std(0) + 1e-10)
        # b = b / (b.std(0) + 1e-10)
        cov = np.mean([
            np.corrcoef(
                shapley_values[key1][i],
                shapley_values[key2][i]
                )[0, 1] for i in range(data_sample_size)
            ])
        covariance_matrix[key1][key2] = cov
covariance_matrix = pd.DataFrame(covariance_matrix)
