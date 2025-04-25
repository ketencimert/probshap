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
if __name__ == '__main__':
    #bike, parkinsons, medical, 
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--dataset', default='medical', type=str)
    parser.add_argument('--fold', default='0', type=str)
    parser.add_argument('--cv_folds', default=5, type=int)

    parser.add_argument('--preprocess', default=True, type=bool)

    parser.add_argument('--exact_model_name', default='XGB', type=str)

    parser.add_argument('--prior', default='masked', type=str)
    parser.add_argument('--estimate', default='TREE', type=str)
    parser.add_argument('--beta', default=1, type=float)
    parser.add_argument('--inc_beta', default=False)
    parser.add_argument('--act', default='elu', type=str)

    args = parser.parse_args()
    os.makedirs('./figures', exist_ok=True)
    
    l2 = dict()
    time_memory = dict()
    SEED = 11
    random.seed(SEED), np.random.seed(SEED), torch.manual_seed(SEED)
    
    datasets = ['medical', 'bike', 'parkinsons']
    
    for dataset in tqdm(datasets):
    
        for fold in tqdm(range(args.cv_folds)):

            tensor_dtype = torch.float
                    
            data, features, dtypes, target_scale, \
                predictive_distribution  = load_dataset(
                dataset=dataset,
                )
            _, _, _, stats = prepare_fold(
                *data[0], 1024, 1, np.asarray([0]),
                args.device, tensor_dtype, dtypes, args.preprocess
                )
    
            x, y, fs, ys, sigmas = data[1]
            x = transform(x, stats)
            x = torch.tensor(x, dtype=tensor_dtype).to(args.device)

            print('Number of features are {}'.format(x.size(-1)))

            ##########################################################################
            data_sample_size = 1024
            sample_sizes = [16, 32, 64, 128]
            x_sample = shap.sample(x, data_sample_size)
            estimates = [
                'EXACT',
                'KERNEL', 
                'FEEDFORWARD', 
                'PERMUTATION', 
                # 'FASTSHAP'
                ]
            ##########################################################################

            exact_model = load_model(
                model_name=args.exact_model_name, 
                prior=None, 
                act=None,
                beta=None, 
                inc_beta=None, 
                fold=fold, 
                dataset=dataset, 
                preprocess=args.preprocess,
                device=None
                )

            model_to_explained = load_model(
                model_name='VariationalShapley', 
                prior=args.prior, 
                act=args.act,
                beta=args.beta, 
                inc_beta=args.inc_beta, 
                fold=fold, 
                dataset=dataset, 
                preprocess=args.preprocess,
                device=args.device
                )

            shapley_values = defaultdict(list)
            for estimate in tqdm(estimates, total=len(estimates)):
                if estimate == 'EXACT':
                    shapley_values[estimate] = get_shapley_values(
                        exact_model, 
                        'TREE',
                        features,
                        dtypes,
                        x_sample,
                        stats
                        ).reshape(-1)
                elif estimate == 'FASTSHAP':
                    starting_time = time.time()
                    shapley_values[estimate] = get_shapley_values(
                            train_fastshap(model_to_explained, x), 
                            estimate,
                            features,
                            dtypes,
                            x_sample,
                            stats
                            ).reshape(-1)
                    time_required = time.time() - starting_time
                    time_memory[(dataset, fold, estimate)] = time_required
                elif estimate == 'FEEDFORWARD':
                    #then we are doing feedforward, no need for sampling
                    starting_time = time.time()
                    shapley_values[estimate] = get_shapley_values(
                            model_to_explained, 
                            estimate,
                            features,
                            dtypes,
                            x_sample,
                            stats
                            ).reshape(-1)
                    time_required = time.time() - starting_time
                    time_memory[(dataset, fold, estimate)] = time_required
                else:
                    #else, we have some kind of sampling parameter
                    for nsamples in sample_sizes:
                        starting_time = time.time()
                        shapley_values[estimate].append(
                            get_shapley_values(
                                model_to_explained, 
                                estimate,
                                features,
                                dtypes,
                                x_sample,
                                stats,
                                nsamples=nsamples,
                                ).reshape(-1, 1)
                            )
                    shapley_values[estimate] = np.concatenate(
                        shapley_values[estimate], -1
                        )
                    time_required = time.time() - starting_time
                    time_memory[(dataset, fold, estimate)] = time_required
            
            os.makedirs('./shapley_values', exist_ok=True)
            with open(
                    './shapley_values/shapley_values_{}_fold_{}.pkl'.format(dataset, fold),
                'wb'
                ) as f:
                pickle.dump(shapley_values,f)
            
            for estimate in shapley_values.keys():
                if estimate != 'EXACT':
                    l2_key = (dataset, fold, estimate)
                    try:
                        l2[l2_key] = (
                            shapley_values[estimate] - shapley_values['EXACT']
                            ) ** 2
                    except:
                        l2[l2_key] = (
                    shapley_values[estimate] - shapley_values['EXACT'].reshape(-1,1)
                            ) ** 2
                    l2[l2_key] = np.nanmean(l2[l2_key], 0) **0.5
