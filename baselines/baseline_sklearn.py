# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 01:39:55 2021

@author: Mert Ketenci
"""
import argparse
import os
from collections import defaultdict
import random

from interpret.glassbox import (
    ExplainableBoostingRegressor,
    ExplainableBoostingClassifier
    )

import lightgbm as lgbm

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import pickle

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    GradientBoostingClassifier,
    RandomForestClassifier
    )
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC, SVR

from xgboost import XGBRegressor, XGBClassifier

from datasets import load_dataset
from utils import pr_auc, rmse, prepare_fold

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune', default=False) #we want true by default to prevent errors
    parser.add_argument('--dataset', default='bike', type=str)
    parser.add_argument('--cv_folds', default=1, type=int)
    parser.add_argument('--preprocess', default=True)
    parser.add_argument('--tensor_dtype', default='float32', type=str)
    parser.add_argument('--model_name', default='RF', type=str)
    args = parser.parse_args()

    SEED = 11
    ITERATIONS = 1
    random.seed(SEED), np.random.seed(SEED), torch.manual_seed(SEED)

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
    
    tensor_dtype = {
        'float64': torch.double,
        'float32': torch.float,
    }[args.tensor_dtype]

    data, features, dtypes, \
        target_scale, predictive_distribution = load_dataset(
        dataset=args.dataset
        )
    x, y = data[0]
    if predictive_distribution == 'Normal':
        task = 'regression'
        metric = rmse
        METRIC = 'RMSE'
    else:
        task = 'classification'
        metric = pr_auc
        METRIC = 'PR-AUC'

    d_in = x.shape[1]

    n = len(x)
    tr_size = int(n * 0.7)

    folds = np.array(list(range(args.cv_folds)) * n)[:n]
    np.random.shuffle(folds)

    if args.model_name == 'RF':
        if task == 'classification':
            Model_ = RandomForestClassifier
            param_grid = {
                'ccp_alpha': [0.0, 1e-1, 1e-2],
                'max_depth': [None, 4, 8, 16, 40, 100],
                'min_samples_leaf': [1, 3, 5, 10],
                'min_samples_split': [2, 4, 6, 12],
                'n_estimators': [100, 200, 600, 800]
                }

        else:
            Model_ = RandomForestRegressor
            param_grid = {
                'ccp_alpha': [0.0, 1e-1, 1e-2],
                'max_depth': [None, 4, 8, 16, 40, 100],
                'min_samples_leaf': [1, 3, 5, 10],
                'min_samples_split': [2, 4, 6, 12],
                'n_estimators': [100, 200, 600, 800]
                }

    elif args.model_name == 'EBM':
        if task == 'classification':
            Model_ = ExplainableBoostingClassifier
            param_grid = {
                'outer_bags': [8, 25, 75, 100],
                'inner_bags': [0, 1, 2, 5, 10]
                }
        else:
            Model_ = ExplainableBoostingRegressor
            param_grid = {
                'outer_bags': [8, 25, 75, 100],
                'inner_bags': [0, 1, 2, 5, 10]
                }

    elif args.model_name == 'GB':
        if task == 'classification':
            Model_ = GradientBoostingClassifier
            param_grid = {
            "learning_rate": [0.01, 0.025, 0.05, 0.075],
            "min_samples_split": [2, 16, 32],
            "min_samples_leaf": [1, 5, 16, 32],
            "max_depth": [3, 4, 8],
            "subsample": [0.5, 0.8, 1.0],
            "n_estimators": [50, 100, 300, 600],
            "ccp_alpha": [0.0, 1e-1, 1e-2]
            }
        else:
            Model_ = GradientBoostingRegressor
            param_grid = {
            "learning_rate": [0.01, 0.025, 0.05, 0.075],
            "min_samples_split": [2, 16, 32],
            "min_samples_leaf": [1, 5, 16, 32],
            "max_depth": [3, 4, 8],
            "subsample": [0.5, 0.8, 1.0],
            "n_estimators": [50, 100, 300, 600],
            "alpha": [9e-1, 9e-2]
            }

    elif args.model_name == 'XGB':
        if task == 'classification':
            Model_ = XGBClassifier
            param_grid = {
                "learning_rate"    : [0.05, 0.10, 0.15, 0.30],
                "max_depth"        : [3, 5, 8, 15],
                "min_child_weight" : [1, 3, 7],
                "gamma"            : [0.0, 0.1, 0.3],
                "colsample_bytree" : [0.3, 0.4, 0.5],
                # 'ccp_alpha': [0.0,1e-3,1e-2],
                # "min_impurity_decrease": [0, 1e-1]
                }
        else:
            Model_ = XGBRegressor
            param_grid = {
    'objective': ['reg:squarederror'],  # For regression tasks
    'eval_metric': ['rmse'],           # Root Mean Squared Error
    'booster': ['gbtree'],             # Tree-based models
    'max_depth': [6],                  # Maximum depth of the trees
    'learning_rate': [0.1],           # Learning rate
    'subsample': [0.8],               # Fraction of samples used for tree building
    'colsample_bytree': [0.9],        # Fraction of features used for tree building
    'n_estimators': [500],             # Number of trees (boosting rounds)
    'min_child_weight': [1],          # Minimum sum of instance weight (hessian) needed in a child
    'gamma': [0],                      # Minimum loss reduction required to make a further partition on a leaf node
    'reg_alpha': [0],                 # L1 regularization term on weights
    'reg_lambda': [1],                # L2 regularization term on weights
    # 'seed': [42]                       # Random seed for reproducibility
}

    elif args.model_name == 'LIN':
        if task == 'classification':
            Model_ = LogisticRegression
            param_grid = {
                'C': [5e-2, 1e-1, 5e-1, 1],
                'max_iter': [10000]
                }
        else:
            Model_ = Ridge
            param_grid = {
                'alpha': [5e-2, 1e-1, 5e-1, 1],
                'max_iter': [10000]
                }

    elif args.model_name == 'LGBM':
        if task == 'classification':
            Model_ = lgbm.sklearn.LGBMClassifier
            param_grid = {
            'num_leaves': [31, 50, 70, 100],    # Number of leaves in full tree
            'max_depth': [-1, 5, 7, 10],        # Maximum tree depth for base learners, <=0 means no limit.
            'learning_rate': [0.001, 0.01, 0.05, 0.1],  # Boosting learning rate
            'n_estimators': [100, 200, 500],    # Number of boosted trees to fit.
            'subsample_for_bin': [200000, 500000],  # Number of samples for constructing bins.
            'min_split_gain': [0.0, 0.1, 0.5],  # Minimum loss reduction required to make a further partition
            'min_child_weight': [1e-3, 1e-2, 1e-1, 1], # Minimum sum of instance weight (hessian) needed in a child (leaf)
            'min_child_samples': [20, 30],      # Minimum number of data needed in a child (leaf)
            'subsample': [0.8, 0.9, 1.0],       # Subsample ratio of the training instance.
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0], # Subsample ratio of columns when constructing each tree.
            'reg_alpha': [0, 1, 2],             # L1 regularization term on weights
            'reg_lambda': [0, 1, 2],            # L2 regularization term on weights
            'boosting_type': ['gbdt', 'dart'],  # Boosting type: gbdt=traditional Gradient Boosting Decision Tree, dart=Dropouts meet Multiple Additive Regression Trees
        }
        else:
            Model_ = lgbm.sklearn.LGBMRegressor
            param_grid = {
            'num_leaves': [31, 50, 70, 100],    # Number of leaves in full tree
            'max_depth': [-1, 5, 7, 10],        # Maximum tree depth for base learners, <=0 means no limit.
            'learning_rate': [0.001, 0.01, 0.05, 0.1],  # Boosting learning rate
            'n_estimators': [100, 200, 500],    # Number of boosted trees to fit.
            'subsample_for_bin': [200000, 500000],  # Number of samples for constructing bins.
            'min_split_gain': [0.0, 0.1, 0.5],  # Minimum loss reduction required to make a further partition
            'min_child_weight': [1e-3, 1e-2, 1e-1, 1], # Minimum sum of instance weight (hessian) needed in a child (leaf)
            'min_child_samples': [20, 30],      # Minimum number of data needed in a child (leaf)
            'subsample': [0.8, 0.9, 1.0],       # Subsample ratio of the training instance.
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0], # Subsample ratio of columns when constructing each tree.
            'reg_alpha': [0, 1, 2],             # L1 regularization term on weights
            'reg_lambda': [0, 1, 2],            # L2 regularization term on weights
            'boosting_type': ['gbdt', 'dart'],  # Boosting type: gbdt=traditional Gradient Boosting Decision Tree, dart=Dropouts meet Multiple Additive Regression Trees
        }
    elif args.model_name == 'SVM':
        if task == 'classification':
            Model_ = SVC
            param_grid = {
            'C':  [5e-2, 1e-1, 5e-1, 1],
            'kernel': ['poly', 'rbf'],
            'degree':[2,3,4],
            'gamma':['scale', 'auto'],
            'probability':[True]
            }
        else:
            Model_ = SVR
            param_grid = {
            'C':  [5e-2, 1e-1, 5e-1, 1],
            'kernel': ['poly', 'rbf'],
            'degree':[2,3,4],
            'gamma':['scale', 'auto'],
            }
    else:
        raise ValueError('Model not implemented.')

    fold_results = defaultdict(lambda: defaultdict(list))

    for fold in tqdm(range(args.cv_folds)):

        tr_dataloader, val_dataloader, te_dataloader, _ = prepare_fold(
            x, y, 1024, fold, folds,
            'cpu', tensor_dtype, dtypes, args.preprocess
            )

        x_tr, y_tr = tr_dataloader.dataset.__getds__()
        x_val, y_val = val_dataloader.dataset.__getds__()
        x_te, y_te = te_dataloader.dataset.__getds__()
        
        model = Model_()

        if args.tune:

            model = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=ITERATIONS,
                )

            model.fit(
                X=np.concatenate([x_tr, x_val],0),
                y=np.concatenate([y_tr, y_val],0)
                )

            best_estimator = model.best_estimator_.get_params()

            if args.model_name == 'ebm':
                best_estimator.pop('feature_types')
                best_estimator.pop('feature_names')

            model = Model_(**best_estimator)

        model.fit(
            X=np.concatenate([x_tr, x_val],0), 
            y=np.concatenate([y_tr, y_val],0)
            )

        if task == 'classification':
            y_pred = model.predict_proba(x_te)[:,1]
        else:
            y_pred = model.predict(x_te)

        fold_results['Fold: {}'.format(fold)][METRIC] = metric(y_pred, y_te)
        
        if args.cv_folds == 1:
            fold = 'X'
        
        os.makedirs('./baseline_checkpoints', exist_ok=True)
        with open(
                './baseline_checkpoints/{}_fold_{}_{}_({}).pkl'.format(
                args.dataset,
                fold,
                args.model_name,
                FLAGS
                ),
                'wb'
                ) as f:
            pickle.dump(model,f)

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
            args.model_name,
            )
        )