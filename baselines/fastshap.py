# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 21:14:07 2023

@author: Mert
"""

import argparse

import numpy as np

from torch import nn
import pickle

import torch
from fastshap.utils import MaskLayer1d, KLDivLoss
from fastshap import Surrogate

import os.path

import os
import random

from datasets import load_dataset


from utils import prepare_fold, load_model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--dataset', default='synthetic1', type=str)

    parser.add_argument('--model_name', default='VariationalShapley', type=str)
    parser.add_argument('--prior', default='masked', type=str)
    parser.add_argument('--inc_beta', default=False)
    parser.add_argument('--beta', default=1, type=float)
    parser.add_argument('--fold', default=0, type=int)

    args = parser.parse_args()
    os.makedirs('./figures', exist_ok=True)

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

    SEED = 11
    random.seed(SEED), np.random.seed(SEED), torch.manual_seed(SEED)

    model, preprocess = load_model(args)
    model.eval()
    tensor_dtype = torch.float

    data, features, dtypes, target_scale, \
        predictive_distribution  = load_dataset(
        dataset=args.dataset,
        )
    tr_dataloader, val_dataloader, te_dataloader, _ = prepare_fold(
        *data[0], 1024, 1, np.asarray([0]),
        args.device, tensor_dtype, dtypes, preprocess
        )
    
    x_tr, y_tr = tr_dataloader.dataset.__getds__()
    x_val, y_val = val_dataloader.dataset.__getds__()
    x_te, y_te = te_dataloader.dataset.__getds__()

    num_features = x_tr.shape[-1]
    # Select device
    device = torch.device('cuda')

    surr = nn.Sequential(
        MaskLayer1d(value=2, append=True),
        nn.Linear(2 * num_features, 128),
        nn.LayerNorm(128),
        nn.ELU(inplace=True),
        nn.Linear(128, 128),
        nn.LayerNorm(128),
        nn.ELU(inplace=True),
        nn.Linear(128, 1)
        ).to(device)

    # Set up surrogate object
    surrogate = Surrogate(surr, num_features)

    # Set up original model
    def original_model(x):
        return model.predict(x, numpy=False).unsqueeze(-1)

    # Train
    surrogate.train_original_model(
        x_tr,
        x_val,
        original_model,
        batch_size=1024,
        max_epochs=1000,
        loss_fn=nn.MSELoss(),
        validation_samples=10,
        validation_batch_size=10000,
        verbose=True
        )

    from fastshap import FastSHAP

    # Create explainer model
    explainer = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(inplace=True),
        nn.LayerNorm(128),
        nn.Linear(128, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, num_features)
        ).to(device)
    
    # Set up FastSHAP object
    fastshap = FastSHAP(
        explainer, 
        surrogate, 
        normalization='additive',
        link=None
        )
    # Train
    fastshap.train(
        x_tr,
        x_val,
        batch_size=1024,
        num_samples=32,
        max_epochs=200,
        validation_samples=128,
        verbose=True
        )
    # Save explainer
    os.makedirs('./baseline_checkpoints', exist_ok=True)
    with open(
            './baseline_checkpoints/{}_fold_{}_{}_({}).pkl'.format(
            args.dataset,
            args.fold,
            'FASTSHAP',
            FLAGS
            ),
            'wb'
            ) as f:
        pickle.dump(fastshap,f)
    
    
    
    fastshap_values = fastshap.shap_values(
        x_te[0].reshape(1,-1)
        )[0].squeeze(-1)
    