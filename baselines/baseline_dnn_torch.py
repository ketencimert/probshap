# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 15:08:39 2023

@author: Mert
"""
import numpy as np
import random

import argparse
import ast

import torch
from torch import nn
from torch import optim

from sklearn.model_selection import ParameterGrid

from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from modules import create_feedforward_layers
from datasets import load_dataset

from tqdm import tqdm

from utils import (
    train_one_epoch,
    evaluate_model,
    cache_epoch_results,
    get_best_model,
    check_early_stop,
    save_epoch_stats,
    prepare_fold,
    cache_fold_results,
    rmse,
    pr_auc
    )

from collections import defaultdict

import os
import pandas as pd

class Model(nn.Module):
    def __init__(
            self, d_in, d_hid, n_layers, act, 
            norm, p, predictive_distribution, **extras
            ):
        super(Model, self).__init__()
        
        self.predictive_distribution = predictive_distribution
        
        self.network = nn.Sequential(
                *create_feedforward_layers(
                d_in, d_hid, 2, n_layers, act, p, norm
                )[0]
                )

    def forward(self, x, y):
        
        loc, scale = self.network(x).split(1, -1)
        loc = loc.squeeze(-1)
        scale = scale.squeeze(-1)
        if self.predictive_distribution == 'Normal':
            #MSE LOSS
            loglikelihood = Normal(
                loc=loc,
                scale=nn.Softplus()(scale),
                ).log_prob(y).mean(0)
            predictions = loc
        elif self.predictive_distribution == 'Bernoulli':
            #BCE LOSS
            loglikelihood = Bernoulli(
                logits=loc,
                ).log_prob(y).mean(0)
            predictions = nn.Sigmoid()(loc)

        return loglikelihood, loglikelihood, predictions
        
def run_model(
        model, best_model, lr, wd,
        tr_dataloader, val_dataloader, metric, criterion,
        epochs
        ):

    optimizer = optim.Adam(model.parameters(), 
        lr=lr,  weight_decay=wd,
        )
    
    epoch_results = defaultdict(list)
    
    STOP = 0
    
    for epoch in tqdm(range(int(epochs)), position=0, leave=True):
        
        tr_elbo, tr_score = train_one_epoch(
            model,
            optimizer,
            tr_dataloader,
            metric
            )
    
        val_elbo, val_score, mse_phi0 = evaluate_model(
            model, val_dataloader, metric
            )
    
        epoch_results = cache_epoch_results(
            epoch_results, tr_elbo, val_elbo,
            tr_score, val_score, criterion, metric,
            mse_phi0
            )
    
        best_model = get_best_model(
            best_model, model, epoch_results, metric
            )
    
        if epoch % args.check_early_stop == 0 and epoch != 0:
            STOP = check_early_stop(
                STOP, epoch, args.check_early_stop,
                epoch_results, metric
                )
        
        elif STOP == args.early_stop:
            print('Early stopping...')
            break

    return best_model, epoch_results

def save_fold_stats(
        fold_results, best_model, dataset, fold,
        metric, MODEL_NAME, FLAGS
        ):

    if metric.__name__ == 'rmse':
        METRIC = 'RMSE'
    else:
        METRIC = 'PR AUC'
    
    if len(fold_results) != 0:
        print('\n' + '-'*26 + 'Results' + '-'*27)
        print('Mean Fold Score : {}'.format(
            np.mean(
                fold_results[
                    'Fold: {}'.format(fold)
                    ][METRIC]
                )
            )
            )
        print('Standard Deviation : {}'.format(
            np.std(
                fold_results[
                    'Fold: {}'.format(fold)
                    ][METRIC]
                )
            )
            )
        print('Fold Scores : {}'.format(
            fold_results[
                'Fold: {}'.format(fold)
                ][METRIC]
            )
            )
        print('-'*60 + '\n')

    os.makedirs('./baseline_checkpoints', exist_ok=True)
    torch.save(
        best_model,
        './baseline_checkpoints/{}_fold_{}_{}_({}).pth'.format(
            dataset,
            fold,
            MODEL_NAME,
            FLAGS
            )
        )

    fold_results = pd.DataFrame(fold_results)
    for key in fold_results.keys():
        fold_results[key] = [
            _[0] for _ in fold_results[key]
        ]
    os.makedirs('./fold_results', exist_ok=True)
    fold_results.to_csv(
        './fold_results/{}_{}_fold_results_({}).csv'.format(
            dataset,
            MODEL_NAME,
            FLAGS
            )
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #device args
    parser.add_argument(
        '--device', default='cuda', type=str,
        help='device to train the model.'
        )

    parser.add_argument('--epochs', default=int(1e5), type=int,
                        help='number of training epochs.'
                        )
    #data, fold, tune, metric args
    parser.add_argument('--cv_folds', default=5, type=int,
                        help='if you want to plot shapes use cv_folds=1'
                        )
    parser.add_argument('--dataset', default='synthetic1', type=str)
    parser.add_argument('--preprocess', default=True,
                        help='convert to action="store_true" if not \
                        running on an IDE.'
                        )
    parser.add_argument('--check_early_stop', default=150, type=int,
                        help='check early stop every 150 epochs.'
                        )
    parser.add_argument('--early_stop', default=30, type=int,
                        help='stop if no improvement for 20 checks.'
                        )
    args = parser.parse_args()

    MODEL_NAME = 'DNN'

    SEED = 11
    random.seed(SEED), np.random.seed(SEED), torch.manual_seed(SEED)

    HYPERPARAMETER_SAMPLES = 500
    TUNE_EPOCHS = args.epochs / 200
    
    data, features, dtypes, target_scale, predictive_distribution\
        = load_dataset(dataset=args.dataset)
    x, y = data[0]

    param_grid = {
            'batch_size': [1024, 512],
            'lr':[5e-4, 1e-3, 2e-3],
            'act': ['relu', 'snake', 'elu'],
            'norm':[None, 'layer', 'batch'],
            'n_layers':[2, 3, 4, 5],
            'd_hid':[25, 50, 75, 100, 200],
            'wd': [0, 1e-10, 1e-8, 1e-6],
            'p': [0, 0.2, 0.4, 0.5],
            }
    param_grid['d_in'] = [x.shape[-1]]
    param_grid['d_out'] = [1]
    param_grid['predictive_distribution'] = [predictive_distribution]
    param_grid = ParameterGrid(param_grid)

    param_grid = [
        param_grid[_] for _ in np.random.choice(
            len(param_grid),
            min(len(param_grid), HYPERPARAMETER_SAMPLES),
            replace=False
        )
    ]

    if predictive_distribution == 'Normal':
        d_out = 1
        criterion = min
        metric = rmse
        METRIC = 'RMSE'
    elif predictive_distribution == 'Bernoulli':
        d_out = 1
        criterion = max
        metric = pr_auc
        METRIC = 'PR AUC'
    
    n = len(x)
    tr_size = int(n * 0.7)
    
    folds = np.array(list(range(args.cv_folds)) * n)[:n]
    np.random.shuffle(folds)
    
    fold_results = defaultdict(lambda: defaultdict(list))
    
    for fold in tqdm(range(args.cv_folds), position=0, leave=True):

        best_param_dict = {}
        
        for param in tqdm(param_grid, total=len(param_grid)):

            tr_dataloader, val_dataloader, te_dataloader, _ = prepare_fold(
                x, y, param['batch_size'], fold, folds,
                args.device, torch.float32, dtypes, args.preprocess
                )

            best_model = None
            model = Model(**param).to(args.device)
            
            best_model, epoch_results = run_model(
                model, best_model, param['lr'], param['wd'],
                tr_dataloader, val_dataloader, metric, criterion,
                TUNE_EPOCHS
                )
            
            best_param_dict[
                criterion(epoch_results['Valid Best {}'.format(METRIC)])
                ] = param

        best_param = best_param_dict[criterion(best_param_dict.keys())]

        best_model = None
        model = Model(**best_param).to(args.device)       
        best_model, epoch_results = run_model(
                model, best_model, param['lr'], param['wd'],
                tr_dataloader, val_dataloader, metric, criterion,
                args.epochs
                )
        
        fold_results = cache_fold_results(
                fold_results, best_model, te_dataloader,
                fold, metric
                )
            
        FLAGS = ', '.join(
            [
                str(y) + ' ' + str(x) for (y,x) in best_param.items() if y not in [
                    'device',
                    'dataset',
                    'early_stop',
                    'check_early_stop',
                    'num_inducing_points',
                    'network',
                    'num_tries',
                    'epochs',
                    'batch_size'
                    ]
                ]
            )
            
        save_fold_stats(
            fold_results, best_model, args.dataset, fold,
            metric, MODEL_NAME, FLAGS
            )