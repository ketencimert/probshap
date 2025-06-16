# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 17:58:28 2022

@author: Mert
"""
from copy import deepcopy
from collections import defaultdict
import os

import numpy as np
import pandas as pd
import random

import pickle

import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from typing import Iterable 

import shap
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from torch.distributions.normal import Normal

import matplotlib
import matplotlib.pyplot as plt


# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Helvetica"
# })

# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'
# matplotlib.rc('text', usetex=True)
# matplotlib.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"

def sample_covariance(x, y):
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    cov = torch.sum((x - x_mean) * (y - y_mean)) / (x.numel() - 1)
    return cov.detach()

class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y, device, dtype=torch.double):
        x = x.astype(np.float32)
        self.ds = [
            [
                torch.tensor(x, dtype=dtype),
                torch.tensor(y, dtype=dtype),
                torch.tensor(i, dtype=torch.long),
            ] for i, (x, y) in enumerate(zip(x, y))
        ]

        self.device = device
        self._cache = dict()

        self.input_size_ = x.shape[1]

    def __getitem__(self, index: int) -> torch.Tensor:

        if index not in self._cache:

            self._cache[index] = list(self.ds[index])

            if 'cuda' in self.device:
                self._cache[index][0] = self._cache[
                    index][0].to(self.device)

                self._cache[index][1] = self._cache[
                    index][1].to(self.device)
                
                self._cache[index][2] = self._cache[
                    index][2].to(self.device)
                                
        return self._cache[index]

    def __len__(self) -> int:

        return len(self.ds)

    def input_size(self):

        return self.input_size_

    def __getds__(self, numpy=True):
        x = torch.cat([ds[0].view(1,-1) for ds in self.ds], axis=0)
        y = torch.stack([ds[1] for ds in self.ds])
        if numpy:
            x = to_np(x)
            y = to_np(y)
        return x, y

def one_hot_encode(dataframe, column):
    categorical = pd.get_dummies(dataframe[column], prefix=column)
    dataframe = dataframe.drop(column, axis=1)
    return pd.concat([dataframe, categorical], axis=1, sort=False)

def to_np(tensor):
    try:
        tensor = torch.detach(tensor).cpu().numpy()
    except:
        tensor = tensor
    return tensor

def transform(x, stats):
    loc, scale = stats
    return (x - loc) / scale

def inverse_transform(x, stats):
    if torch.is_tensor(x):
        x = to_np(x)
    loc, scale = stats
    return x * scale + loc

def rmse(y_pred, y):
    score = np.mean((y_pred - y)**2)**0.5
    return score

def pr_auc(y_pred, y):
    try:
        y_pred = y_pred.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
    except:
        pass
    try:
        score = average_precision_score(y, y_pred)
    except:
        score = np.asarray([np.nan])
    return score

def prepare_fold(
        x, y, batch_size, fold, folds, device, tensor_dtype, dtypes, preprocess
        ):

    print('\nPreparing fold : {}'.format(fold))

    if folds.sum() != 0:

        x_split = x[folds != fold]
        y_split = y[folds != fold]

        tr_size = int(x_split.shape[0] * 0.8)

        x_tr, x_val = x_split[:tr_size], x_split[tr_size:]
        y_tr, y_val = y_split[:tr_size], y_split[tr_size:]

        x_te = x[folds == fold]
        y_te = y[folds == fold]

    else:

        tr_size = int(x.shape[0] * 0.8)

        x_tr, x_val = x[:tr_size], x[tr_size:]
        y_tr, y_val = y[:tr_size], y[tr_size:]

        x_te, y_te = x_val, y_val

    loc = np.asarray(
        [
        x_tr[
            :,i
        ].mean(0) if dtypes[i]!='uint8' and preprocess else 0 for i in range(
                len(
                    dtypes
                    )
                )
                ]
                )

    scale = np.asarray(
        [
        x_tr[
            :,i
        ].std(0) + 1 if dtypes[i]!='uint8' and preprocess else 1 for i in range(
                len(
                    dtypes
                    )
                )
                ]
                )
    stats = np.concatenate([loc.reshape(1,-1), scale.reshape(1,-1)], 0)
    x_tr, x_val, x_te = [
        transform(x, stats) for x in [x_tr, x_val, x_te]
        ]

    train_data = Dataset(
        x_tr, y_tr, device, tensor_dtype
    )
    valid_data = Dataset(
        x_val, y_val, device, tensor_dtype
    )
    test_data = Dataset(
        x_te, y_te, device, tensor_dtype
    )

    tr_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        valid_data, batch_size=batch_size, shuffle=False
    )
    te_dataloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False
    )

    return tr_dataloader, val_dataloader, te_dataloader, stats

def get_best_model(best_model, model, epoch_results, metric):
    if metric.__name__ == 'rmse':
        METRIC = 'RMSE'
    else:
        METRIC = 'PR AUC'
    if METRIC == 'RMSE':
        if epoch_results[
                'Valid {}'.format(METRIC)
                ][-1] <= min(
                    epoch_results['Valid {}'.format(METRIC)
                                  ]
                    ):
            best_model = deepcopy(model)
    else:
        if epoch_results[
                'Valid {}'.format(METRIC)
                ][-1] >= max(
                    epoch_results['Valid {}'.format(METRIC)
                                  ]
                    ):
            best_model = deepcopy(model)

    return best_model

def check_early_stop(
        stop, epoch, check_early_stop, epoch_results, metric
        ):

    if metric.__name__ == 'rmse':
        METRIC = 'RMSE'
    else:
        METRIC = 'PR AUC'
    
    print('\n\nValidating\
        \nELBO : {:.2E} | Current Epoch Score : {:.3f} \
        \nEpoch Rolling Mean Score : {:.3f} | Best Epoch Score : {:.3f} |\
        \n\n'
        .format(
            epoch_results['Valid ELBO'][-1],
            epoch_results['Valid {}'.format(METRIC)][-1],
            epoch_results['Valid Mean {}'.format(METRIC)][-1],
            epoch_results['Valid Best {}'.format(METRIC)][-1],
            ),
        end="\r"
        )
    if epoch > check_early_stop:
        if METRIC == 'RMSE':
            if epoch_results[
                'Valid Best {}'.format(METRIC)
                ][-1] >= epoch_results[
                    'Valid Best {}'.format(METRIC)
                    ][-check_early_stop]:
                stop += 1
            else:
                stop = 0
        elif METRIC == 'PR AUC':
            if epoch_results[
                'Valid Best {}'.format(METRIC)
                ][-1] <= epoch_results[
                    'Valid Best {}'.format(METRIC)
                    ][-check_early_stop]:
                stop += 1
            else:
                stop = 0

        print('STOP : {}'.format(stop))
    return stop

def cache_epoch_results(
        epoch_results, tr_elbo, val_elbo,
        tr_score, val_score, criterion, metric, mse_phi0
        ):

    if metric.__name__ == 'rmse':
        METRIC = 'RMSE'
    else:
        METRIC = 'PR AUC'
    
    epoch_results['Train ELBO'].append(tr_elbo)
    epoch_results['Valid ELBO'].append(val_elbo)
    epoch_results['Train {}'.format(METRIC)].append(tr_score)
    epoch_results['Valid {}'.format(METRIC)].append(val_score)
    epoch_results['Valid MSE PHI0'].append(mse_phi0)

    epoch_results['Valid Best {}'.format(METRIC)].append(
        criterion(epoch_results['Valid {}'.format(METRIC)])
        )
    epoch_results['Valid Mean {}'.format(METRIC)].append(
        np.mean(epoch_results['Valid {}'.format(METRIC)][-10:])
        )
    epoch_results['Valid Mean ELBO'].append(
        np.mean(epoch_results['Valid ELBO'][-10:])
        )
    epoch_results['Valid Mean MSE PHI0'].append(
        np.mean(epoch_results['Valid MSE PHI0'][-10:])
        )

    return epoch_results

def save_epoch_stats(
        model,
        epoch_results, dataset, fold,
        MODEL_NAME, FLAGS, metric,
        warm_up=0
        ):

    if metric.__name__ == 'rmse':
        METRIC = 'RMSE'
    else:
        METRIC = 'PR AUC'
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    ax[0].plot(
        epoch_results['Train ELBO'][warm_up:], color='b', label='Train'
        )
    ax[0].plot(
        epoch_results['Valid ELBO'][warm_up:], color='r', label='Valid'
        )

    # label = '$\mathcal{L}(\mathbf{X},\mathbf{y},\Psi)$'
    label = 'Loss'
    ax[0].set_ylabel(
        label,
        size=25
        )
    ax[0].set_xlabel(
        'Epochs',
        size=25
        )
    ax[0].tick_params(axis='both', labelsize=20)
    # ax[0].tick_params(axis='both', which='minor',labelsize=20)

    color = ['b', 'r']
    i = 0
    for (key, value) in epoch_results.items():
        if METRIC in key:
            if ('Best' not in key) and ('Mean' not in key):
                ax[1].plot(value[warm_up:], color=color[i], label=key)
                i += 1

    ax[1].set_ylabel('Performance Metric ({})'.format(METRIC),
                     size=25
                     )
    ax[1].set_xlabel(
        'Epochs',
        size=25
        )
    ax[1].tick_params(axis='both', labelsize=20)
    # ax[1].tick_params(axis='both', which='minor',labelsize=20)

    plt.tight_layout()

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.20,1), prop={'size': 25})

    os.makedirs('./fold_figures', exist_ok=True)
    plt.savefig("./fold_figures/{}_fold_{}_{}_figs_({}).svg".format(
            dataset,
            fold,
            MODEL_NAME,
            FLAGS
            ),
        bbox_inches='tight'
        )

    epoch_results = pd.DataFrame(epoch_results)
    os.makedirs('./epoch_results', exist_ok=True)
    epoch_results.to_csv(
        './epoch_results/{}_fold_{}_{}_epoch_res_({}).csv'.format(
            dataset,
            fold,
            MODEL_NAME,
            FLAGS
            )
        )

def cache_fold_results(
        fold_results, best_model, te_dataloader, fold, metric
        ):

    if metric.__name__ == 'rmse':
        METRIC = 'RMSE'
    else:
        METRIC = 'PR AUC'
    
    elbo, score, mse_phi0 = evaluate_model(
        best_model.eval(),
        te_dataloader,
        metric
        )

    fold_results['Fold: {}'.format(fold)][METRIC].append(score)

    print('\n\n'+'-'*80)
    print(
        'Testing Fold : {} | ELBO : {:.2E} | Test Score : {:.3f} | \n'
        .format(fold, elbo, score), end="\r"
          )
    print('-'*80)

    return fold_results

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

    os.makedirs('./model_checkpoints', exist_ok=True)
    torch.save(
        best_model,
        './model_checkpoints/{}_fold_{}_{}_({}).pth'.format(
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
        
def train_one_epoch(model, optimizer, dataloader, metric):
    tr_loss = []
    y_pred_list = []
    y_list = []
    phi_mean_list = []
    
    model.train()
    for x_tr, y_tr, i_tr in dataloader:
        
        optimizer.zero_grad()
        elbo, loss, proxy_kld, y_pred, phi_mean = model(x_tr, y_tr, i_tr)
        (-loss).backward()
        try:
            model.beta.grad.data =- model.beta.grad.data
        except:
            pass
        # print(model.beta.grad.data)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        tr_loss.append(loss.item())
        y_pred_list.append(y_pred)
        y_list.append(y_tr)
        phi_mean_list.append(phi_mean)
        
    y_pred = to_np(torch.cat(y_pred_list, 0))
    y_tr = to_np(torch.cat(y_list, 0))
    tr_score = metric(y_pred, y_tr)
    tr_loss = np.mean(tr_loss)
    phi_mean = torch.stack(phi_mean_list).mean().item()
    return tr_loss, tr_score, phi_mean

def evaluate_model(
        model, dataloader, metric,
        ):
    
    model.eval()
    with torch.no_grad():
        loss_list = []
        y_pred_list = []
        y_list = []
        phi_mean_list = []
        for x, y, i in dataloader:
            elbo, loss, proxy_kld, y_pred, phi_mean = model(x, y, i)
            # print(proxy_kld)
            loss_list.append(elbo)
            y_pred_list.append(y_pred)
            y_list.append(y)
            phi_mean_list.append(phi_mean)
            
        y_pred = to_np(torch.cat(y_pred_list, 0))
        y = to_np(torch.cat(y_list, 0))
        score = metric(y_pred, y)
        loss = torch.stack(loss_list).mean().item()
        phi_mean = torch.stack(phi_mean_list).mean().item()

    return loss, score, phi_mean

def accuracy(y_pred, y):
    return sum(y_pred == y) / y.shape[0]

def shuffle(x):
    random.shuffle(x)
    return x

def flatten(items):
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x

def create_array(n, k):
    return [(i % k == 0) for i in range(n)]

def find_pkl_files_in_directory(path):
    pkl_files = []

    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.pkl'):
                full_path = os.path.join(dirpath, filename)
                pkl_files.append(full_path)
    return pkl_files

def find_csv_files_in_directory(path):
    csv_files = []

    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.csv'):
                full_path = os.path.join(dirpath, filename)
                csv_files.append(full_path)

    return csv_files

def find_pth_files_in_directory(path):
    pth_files = []

    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.pth'):
                full_path = os.path.join(dirpath, filename)
                pth_files.append(full_path)

    return pth_files

def dataset_in_csv(x, datasets):
    output = False
    for dataset in datasets:
        if dataset in x:
            output = True
    return output

def get_idxs(y_pred, y):
    true_negative = np.argmax((y_pred < 0.5) * (y == 0))
    false_negative = np.argmax((y_pred < 0.5) * (y == 1))
    false_positive = np.argmax((y_pred >= 0.5) * (y == 0))
    true_positive = np.argmax((y_pred >= 0.5) * (y == 1))
    return [true_negative, false_negative, false_positive, true_positive]

def get_importances(
        model, estimate, features, dtypes, x, stats, nsamples=20
        ):
    with torch.no_grad():
        shapley_importances = defaultdict(int)
        write_shapley = True
        if (model.__class__.__name__ == 'Model') and (
                estimate == 'FEEDFORWARD' or estimate == 'MONTECARLO'
                ):
            write_shapley = False
            model = model.double()
            x = x.double()
            shapley_importances = model.feature_importances(
                x,
                stats,
                features,
                dtypes,
                estimate=estimate
                )
        elif estimate == 'TREE':
            cond1 = (
                model.__class__.__name__ == 'LGBMRegressor'
                ) or (model.__class__.__name__ == 'LGBMClassifier')
            cond2 = (
                model.__class__.__name__ == 'RandomForestRegressor'
                ) or (model.__class__.__name__ == 'RandomForestClassifier')
            cond3 = (
                model.__class__.__name__ == 'XGBRegressor'
                ) or (model.__class__.__name__ == 'XGBClassifier')
            if cond1 or cond2 or cond3:
                x = to_np(x)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(x)
            else:
                raise(ValueError)
        elif estimate == 'KERNEL':
            model = model.double()
            x = x.double()
            x = to_np(x)
            explainer = shap.KernelExplainer(
                model.predict, 
                shap.sample(x, nsamples),
                algorithm='kernel'
                )
            shap_values = explainer.shap_values(x, nsamples=nsamples)
        elif estimate == 'PERMUTATION':
            model = model.double()
            x = x.double()
            x = to_np(x)
            explainer = shap.PermutationExplainer(
                model.predict, 
                shap.sample(x, nsamples)
                )
            shap_values = explainer.shap_values(
                x,
                npermutations=nsamples
                )
            shap_values = np.asarray(shap_values, dtype=np.float32)
        elif estimate == 'SAMPLING':
            model = model.double()
            x = x.double()
            x = to_np(x)
            explainer = shap.SamplingExplainer(
                model.predict, 
                shap.sample(x, nsamples)
                )
            shap_values = explainer.shap_values(x, nsamples=nsamples)
            shap_values = np.asarray(shap_values, dtype=np.float32)
        elif estimate == 'EXACT':
            model = model.double()
            x = x.double()
            x = to_np(x)
            explainer = shap.explainers.Exact(
                model.predict, 
                x,
                )
            shap_values = explainer(x, max_evals=131072)
            shap_values = np.asarray(shap_values.values, dtype=np.float32)
            
        if write_shapley:
            for i,(f,d) in enumerate(zip(features, dtypes)):
                idxs = x[:, i].argsort()
                x_ = inverse_transform(x, stats)[:,i][idxs].reshape(-1,1)
                y = shap_values[:,i][idxs].reshape(-1,1)
                err = np.asarray([np.nan] * y.shape[0]).reshape(-1,1)
                shapley_importances[(f, d)] = np.concatenate(
                    [x_, y, err], -1
                    )
    return shapley_importances

def load_model(
        model_name, prior, act, beta, inc_beta, fold, dataset, preprocess, device
        ):
    
    if 'VariationalShapley' in model_name:
        saved_models = find_pth_files_in_directory(
            './model_checkpoints'
            )
        saved_models = [sm  for sm in saved_models if dataset in sm]
        saved_models = [
            sm for sm in saved_models if 'beta {}'.format(beta) in sm
            ]
        saved_models = [
            sm for sm in saved_models if (
                'inc_beta {}'.format(inc_beta) in sm) and (prior in sm)
            ]
        saved_models = [
            sm  for sm in saved_models if 'fold_{}'.format(fold) in sm
            ]
        saved_models = [sm  for sm in saved_models if act in sm]
        saved_models = [
            sm  for sm in saved_models if 'preprocess {}'.format(
                str(preprocess)
                ) in sm
            ]
        saved_models = [sm  for sm in saved_models if model_name in sm]
        print(
            'There is/are {} models that fit the parameters.'.format(
                len(saved_models)
                )
            )
        saved_model = saved_models[0]
        print(saved_model)
        model = torch.load(saved_model, map_location=device)
        model.eval()
    else:
        saved_models = os.listdir('./baseline_checkpoints')
        saved_models = [
            sm for sm in [
                sm  for sm in saved_models if dataset in sm
                ] if model_name in sm
            ]
        saved_models = [
            sm for sm in saved_models if 'fold_{}'.format(fold) in sm
            ]
        saved_models = [
            sm  for sm in saved_models if 'preprocess {}'.format(
                str(preprocess)
                ) in sm
            ]
        print(
            'There is/are {} models that fit the parameters.'.format(
                len(saved_models)
                )
            )
        saved_model = saved_models[0]
        
        with open('./baseline_checkpoints/' + saved_model, 'rb') as f:
            model = pickle.load(f)
    return model

def get_shapley_values(
        model, estimate, features, dtypes, x, stats, nsamples=10
        ):
    with torch.no_grad():
        if (model.__class__.__name__ == 'Model') and (
                estimate == 'FEEDFORWARD' or estimate == 'MONTECARLO'
                ):
            # model = model.double()
            # x = x.double()
            shap_values = model.shapley_values(
                x, estimate=estimate, sample_size=nsamples
                )
            # model = model.float()
            x = x.float()
        elif estimate == 'TREE':
            cond1 = model.__class__.__name__ == 'LGBMRegressor'
            cond2 = model.__class__.__name__ == 'RandomForestRegressor'
            cond3 = model.__class__.__name__ == 'XGBRegressor'
            cond4 = model.__class__.__name__ == 'ExplainableBoostingRegressor'
            if cond1 or cond2 or cond3 or cond4:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(to_np(x))
            else:
                raise(ValueError)
        elif estimate == 'KERNEL':
            # model = model.double()
            # x = x.double()
            explainer = shap.KernelExplainer(
                model.predict, 
                shap.sample(to_np(x), nsamples),
                algorithm='auto',
                )
            shap_values = explainer.shap_values(
                to_np(x), nsamples=nsamples
                )
            # model = model.float()
            x = x.float()
        elif estimate == 'PERMUTATION':
            model = model.double()
            # x = x.double()
            explainer = shap.PermutationExplainer(
                model.predict, 
                shap.sample(to_np(x), nsamples)
                )
            shap_values = explainer.shap_values(
                to_np(x),
                npermutations=nsamples
                )
            shap_values = np.asarray(shap_values, dtype=np.float32)
            model = model.float()
            x = x.float()
        elif estimate == 'FASTSHAP':
            shap_values = []
            for i in range(x.shape[0]):
                shap_values.append(
                model.shap_values(to_np(x)[i].reshape(1,-1)).reshape(1,-1)
                )
            shap_values = np.concatenate(shap_values, 0)
    return shap_values

def train_fastshap(model, x, lr=1e-3):
    
    from torch import nn
    from fastshap.utils import MaskLayer1d
    from fastshap import Surrogate
    
    device = model.prior.combine_loc.weight.device
    dtype = model.prior.combine_loc.weight.dtype
    num_features = x.size(-1)
    surr = nn.Sequential(
        MaskLayer1d(value=2, append=True),
        nn.Linear(2 * num_features, 128, dtype=dtype),
        nn.LayerNorm(128, dtype=dtype),
        nn.ELU(inplace=True),
        nn.Linear(128, 128),
        nn.LayerNorm(128),
        nn.ELU(inplace=True),
        nn.Linear(128, 1, dtype=dtype)
        ).to(device)
    # Set up surrogate object
    surrogate = Surrogate(surr, num_features)

    # Set up original model
    def original_model(x):
        return model.predict(x, numpy=False).unsqueeze(-1)
    # Train
    surrogate.train_original_model(
        to_np(x),
        to_np(x),
        original_model,
        lr=lr,
        batch_size=1024,
        max_epochs=10000,
        loss_fn=nn.MSELoss(),
        validation_samples=10,
        validation_batch_size=10000,
        verbose=True
        )
    from fastshap import FastSHAP
    # Create explainer model
    explainer = nn.Sequential(
        nn.Linear(num_features, 128, dtype=dtype),
        nn.ReLU(inplace=True),
        nn.LayerNorm(128),
        nn.Linear(128, 128, dtype=dtype),
        nn.ReLU(inplace=True),
        nn.Linear(128, num_features, dtype=dtype)
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
        to_np(x),
        to_np(x),
        lr=lr,
        batch_size=1024,
        num_samples=32,
        max_epochs=200,
        validation_samples=128,
        verbose=True
        )
    # Save explainer
    return fastshap
    
