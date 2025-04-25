# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 12:55:11 2021

@author: Mert Ketenci
"""
import argparse
from copy import deepcopy
from collections import defaultdict
import os
import random

import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from datasets import load_dataset
from utils import prepare_fold

from baselines.tune.tune_dnn_keras import build_model, tune_dnn

def main(args):

    tensor_dtype = {
        'float64': torch.double,
        'float32': torch.float,
    }[args.tensor_dtype]

    data, features, dtypes, target_scale, predictive_distribution\
        = load_dataset(dataset=args.dataset)
        
    if predictive_distribution == 'Normal':
        task = 'regression'
    else:
        task = 'classification'
    
    x,y = data[0]
    d_in = x.shape[1]
    n = len(x)

    folds = np.array(list(range(args.cv_folds)) * n)[:n]
    np.random.shuffle(folds)

    fold_results = defaultdict(lambda: defaultdict(list))

    for fold in tqdm(range(args.cv_folds)):
        tr_dataloader, val_dataloader, te_dataloader, _ = prepare_fold(
            x, y, args.batch_size, fold, folds,
            'cpu', tensor_dtype, dtypes, args.preprocess
            )

        if args.tune:
            args = deepcopy(
                tune_dnn(args, tr_dataloader, val_dataloader, task)
                )

        model = build_model(
            d_in, args.layer_size, args.activation, args.batchnorm,
            args.layernorm, args.dropout_input, args.dropout_intermediate,
            task
            )

        if task=='regression':
            loss = keras.losses.MeanSquaredError()
            metric = keras.metrics.RootMeanSquaredError()
            monitor = 'val_root_mean_squared_error'
            mode = 'min'

        else:
            loss = keras.losses.BinaryCrossentropy()
            metric = tf.keras.metrics.AUC(curve='PR')
            monitor = 'auc'
            mode = 'max'

        checkpoint_filepath = './mdl_wts_{}.hdf5'.format(args.dataset)
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            save_best_only=True,
            monitor=monitor,
            mode=mode
            )
            ]

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.lr),
            loss=loss,
            metrics=[metric],
        )

        print("Fit model on training data")

        x_tr, y_tr = tr_dataloader.dataset.__getds__()
        x_val, y_val = val_dataloader.dataset.__getds__()
        x_te, y_te = te_dataloader.dataset.__getds__()

        model.fit(
            x_tr,
            y_tr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        model.load_weights(checkpoint_filepath)
        results = model.evaluate(
            x_te,
            y_te,
            batch_size=128
            )

        fold_results['Fold: {}'.format(fold)][monitor.upper()] = results[-1]

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

        os.makedirs('./baseline_checkpoints', exist_ok=True)
        model.save(
            './baseline_checkpoints/{}_fold_{}_{}_({}).h5'.format(
                args.dataset,
                fold,
                MODEL_NAME,
                FLAGS
                )
            )

    return fold_results

for dataset in [
        'synthetic1', 'synthetic2', 'synthetic3',
        'medical', 'icu', 'bike', 'spambase', 'parkinsons', 'fico', 'adult', 'tamielectric'
        ]:

    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        #tune args
        parser.add_argument('--tune', default=True) #we want true by default to prevent errors
        #training args
        parser.add_argument('--cv_folds', default=5, type=int)
        parser.add_argument('--preprocess', default=True) #we want true by default to prevent errors
        
        parser.add_argument('--dataset', default='synthetic1' ,type=str)
        parser.add_argument('--tensor_dtype', default='float32' ,type=str)
        parser.add_argument('--batch_size', default=1024, type=int)
        parser.add_argument('--epochs', default=20000, type=int)
        parser.add_argument('--lr', default=1e-3, type=int)
        #model args
        parser.add_argument('--dropout_input', default=0.0, type=float)
        parser.add_argument('--dropout_intermediate', default=0, type=float)
        parser.add_argument('--layernorm', default=True)
        parser.add_argument('--batchnorm', action='store_true')
        parser.add_argument('--activation', default='elu' ,type=str)
        parser.add_argument('--layer_size', default=[32, 32], type=int, nargs='+')
        args = parser.parse_args()
    
        SEED = 11
        MODEL_NAME = 'DNN'
        random.seed(SEED), np.random.seed(SEED), torch.manual_seed(SEED)
        tf.random.set_seed(
            SEED
        )
        
        args.dataset = dataset
        
        fold_results = main(args)
        fold_results = pd.DataFrame(fold_results)
        print('\n' + '-'*26 + 'Results' + '-'*27)
        print('Mean Fold Score : {}'.format(
            np.mean(
                fold_results.values
                )
            )
            )
        print('Standard Deviation : {}'.format(
            np.std(
                fold_results.values
                ) / fold_results.shape[1]**0.5
            )
            )
        print('-'*60 + '\n')
        os.makedirs('./fold_results', exist_ok=True)
        fold_results.to_csv(
            './fold_results/{}_{}_fold_results.csv'.format(
                args.dataset,
                MODEL_NAME,
                )
            )
