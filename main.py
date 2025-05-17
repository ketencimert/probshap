# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 17:58:12 2022

@author: Mert
"""
import scipy

import argparse
import random
from collections import defaultdict
import os


import numpy as np
import torch
from torch import optim

from tqdm import tqdm
# from model import Model
from model_meanfield2 import Model
from datasets import load_dataset
from utils import (
    train_one_epoch,
    evaluate_model,
    cache_epoch_results,
    get_best_model,
    check_early_stop,
    save_epoch_stats,
    prepare_fold,
    cache_fold_results,
    save_fold_stats,
    rmse,
    pr_auc,
    accuracy
    )

from utils import to_np

import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #device args
    parser.add_argument(
        '--device', default='cuda', type=str,
        help='device to train the model.'
        )
    #optimization args
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate.'
                        )
    parser.add_argument('--wd', default=0, type=float,
                        help='weight decay.'
                        )
    parser.add_argument('--epochs', default=int(1e5), type=int,
                        help='number of training epochs.'
                        )
    parser.add_argument('--batch_size', default=1024, type=int,
                        help='batch size.'
                        )
    parser.add_argument('--d_emb', default=50, type=int)
    parser.add_argument('--d_hid', default=300, type=int)
    #looks like the more n_layers you have the better approximation
    parser.add_argument('--n_layers', default=4, type=int)
    parser.add_argument('--act', default='elu', type=str)
    parser.add_argument('--norm', default='layer', type=str)
    parser.add_argument('--prior_net', default='masked', type=str)

    parser.add_argument('--p', default=0, type=float)
    parser.add_argument('--beta', default=2, type=float)
    #data, fold, tune, metric args
    parser.add_argument('--cv_folds', default=1, type=int,
                        help='if you want to plot shapes use cv_folds=1'
                        )
    parser.add_argument('--dataset', default='icu', type=str)
    parser.add_argument('--preprocess', default=True,
                        help='convert to action="store_true" if not \
                        running on an IDE.'
                        )
    parser.add_argument('--check_early_stop', default=150, type=int,
                        help='check early stop every 150 epochs.'
                        )
    parser.add_argument('--early_stop', default=200, type=int,
                        help='stop if no improvement for 20 checks.'
                        )
    parser.add_argument('--var', default=100, type=int,
                        help='stop if no improvement for 20 checks.'
                        )
    args = parser.parse_args()

    MODEL_NAME = 'ProbabilisticShapley'

    SEED = 11
    random.seed(SEED), np.random.seed(SEED), torch.manual_seed(SEED)

    ratio = (1/args.beta)**(100*args.check_early_stop/args.epochs)

    FLAGS = ', '.join(
        [
            str(y) + ' ' + str(x) for (y,x) in vars(args).items() if y not in [
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

    fold_results = defaultdict(lambda: defaultdict(list))

    data, features, dtypes, target_scale, predictive_distribution\
        = load_dataset(dataset=args.dataset)
    x, y = data[0]
    d_in = x.shape[-1]

    if predictive_distribution == 'Normal':
        d_out = 1
        criterion = min
        metric = rmse

    elif predictive_distribution == 'Bernoulli':
        d_out = 1
        criterion = max
        metric = pr_auc

    d_in = x.shape[1]
    n = len(x)
    tr_size = int(n * 0.7)

    folds = np.array(list(range(args.cv_folds)) * n)[:n]
    np.random.shuffle(folds)

    for fold in tqdm(range(args.cv_folds), position=0, leave=True):

        try:

            STOP = 0

            tr_dataloader, val_dataloader, te_dataloader, stats = prepare_fold(
                x, y, args.batch_size, fold, folds,
                args.device, torch.float32, dtypes, args.preprocess
                )
            print(stats)
            best_model = None
            model = Model(
                d_in, args.d_hid, d_out, args.d_emb,
                tr_dataloader.dataset.__len__(),
                args.n_layers, args.act, args.norm, args.p, args.beta,
                predictive_distribution,
                phi_net=args.prior_net,
                variance=args.var
                ).to(args.device)
            optimizer = optim.Adam(model.parameters(),
                lr=args.lr,  weight_decay=args.wd,
                )

            epoch_results = defaultdict(list)

            if args.cv_folds == 1:
                fold = 'X'

            for epoch in tqdm(range(args.epochs), position=0, leave=True):

                tr_elbo, tr_score, mse_phi0 = train_one_epoch(
                    model,
                    optimizer,
                    tr_dataloader,
                    metric
                    )

                val_elbo, val_score, _ = evaluate_model(
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

            save_epoch_stats(
                model,
                epoch_results,
                args.dataset,
                fold,
                MODEL_NAME, FLAGS, metric,
                warm_up=0
                )

            fold_results = cache_fold_results(
                fold_results, best_model, te_dataloader,
                fold, metric
                )

        except KeyboardInterrupt:

            print('\nKeyboardInterrupt. Saving the results.')
            save_epoch_stats(
                model,
                epoch_results,
                args.dataset,
                fold,
                MODEL_NAME, FLAGS, metric,
                warm_up=0
                )

            fold_results = cache_fold_results(
                fold_results, best_model, te_dataloader,
                fold, metric
                )

        save_fold_stats(
            fold_results, best_model, args.dataset, fold,
            metric, MODEL_NAME, FLAGS
            )
