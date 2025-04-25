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
from analysis.plot_utils_fig4 import (
    plot_1d,
    plot_2d,
    plot_bar,
    plot_predictive
    )

from utils import (
    transform, prepare_fold, to_np, get_idxs, get_importances, load_model
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--dataset', default='icu', type=str)
    parser.add_argument('--fold', default='0', type=str)
    parser.add_argument('--preprocess', default=True, type=bool)

    parser.add_argument('--row_size', default=1, type=int)

    parser.add_argument('--model_name', default='XGB', type=str)
    parser.add_argument('--prior', default='masked', type=str)
    parser.add_argument('--estimate', default='TREE', type=str)
    parser.add_argument('--beta', default=1, type=float)
    parser.add_argument('--inc_beta', default=False)
    parser.add_argument('--act', default='elu', type=str)

    parser.add_argument('--deviation', default=2.5, type=int)
    parser.add_argument('--min_knot_size', default=20, type=int)
    parser.add_argument('--gf_sigma', default=1, type=int)
    parser.add_argument('--deg', default=1, type=int)

    parser.add_argument('--labelsize', default=15, type=int)

    parser.add_argument('--rotate', default=True)
    parser.add_argument('--scale_of_mean', default=False)
    parser.add_argument('--show_x_ticks', default=True)

    args = parser.parse_args()
    os.makedirs('./figures', exist_ok=True)

    SEED = 11
    random.seed(SEED), np.random.seed(SEED), torch.manual_seed(SEED)

    model = load_model(
        model_name=args.model_name, 
        prior=args.prior, 
        act=args.act,
        beta=args.beta, 
        inc_beta=args.inc_beta, 
        fold=args.fold, 
        dataset=args.dataset, 
        preprocess=args.preprocess,
        device=args.device
        )

    tensor_dtype = torch.float
            
    data, features, dtypes, target_scale, \
        predictive_distribution  = load_dataset(
        dataset=args.dataset,
        )
    _, _, _, stats = prepare_fold(
        *data[0], 1024, 1, np.asarray([0]),
        args.device, tensor_dtype, dtypes, args.preprocess
        )

    x, y, fs, ys, sigmas = data[1]
    x = transform(x, stats)
    x = torch.tensor(x, dtype=tensor_dtype).to(args.device)
    
    shapley_importances = get_importances(
        model, 
        args.estimate,
        features,
        dtypes,
        x,
        stats
        )
        
    feature_size = len(shapley_importances)
    column_size = int(np.ceil(feature_size / args.row_size))
    gridspec_dim = {'nrows': args.row_size, 'ncols': column_size}
    figsize = [12, 2.5]

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(figure=fig, **gridspec_dim)

    row_idx, col_idx = 0, 0
    for ii, (key, value) in enumerate(shapley_importances.items()):
        if not isinstance(key[0],tuple):
            ax_dict = {}
            gs_sub = gridspec.GridSpecFromSubplotSpec(
                nrows=1,
                ncols=1,
                subplot_spec=gs[row_idx, col_idx],
                hspace=0.1,
                # height_ratios=[1.0, 1.0, 1.0 / 6.0]
                # if combine == 'both'
                # else [1.0, 1.0 / 3.0],
            )
            ax_dict['importance'] = fig.add_subplot(gs_sub[0, 0])
            # ax_dict['data_count'] = fig.add_subplot(gs_sub[1, 0])

            ax = plot_1d(
                feature_name=key[0],
                ax_dict=ax_dict,
                feature_dict=value,
                deviation=args.deviation,
                gf_sigma=args.gf_sigma,
                fs_=fs[ii],
                labelsize=args.labelsize,
                deg=args.deg,
                min_knot_size=args.min_knot_size,
                ys=ys[ii]
                )
            if ii==1:
                ax['importance'].legend(
                    loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=4,
                    fontsize=15
                    )
        else:
            ax_dict = {}
            gs_sub = gridspec.GridSpecFromSubplotSpec(
                        nrows=1,
                        ncols=2,
                        subplot_spec=gs[row_idx, col_idx],
                        wspace=0.7,
                        width_ratios=[1.0, 1.0],
                        height_ratios=[1/3]
                    )
            ax_dict['importance'] = fig.add_subplot(gs_sub[0, 0])
            ax_dict['real'] = fig.add_subplot(gs_sub[0, 1])

            plot_2d(
                feature_name=key[0],
                ax_dict=ax_dict,
                feature_dict=value,
                fs_=fs[ii],
                labelsize=args.labelsize
                )

        # if synthetic:
        #     handles, labels = ax['importance'].get_legend_handles_labels()
        #     fig.legend(
        #         handles, 
        #         labels, 
        #         loc='upper center',
        #         ncol=3, 
        #         labelspacing=0,
        #         prop={'size': 20},
        #         bbox_to_anchor=(0.5, 0.55, 0.0, 0.5)
        #         )
        # names = [r"$\mu_{\theta_1}$", r"$\mu_{\theta_2}$", r"$\mu_{\theta_3}$"]
        # if col_idx == 0 and not isinstance(key[0], tuple):
        # ax_dict['importance'].set_ylabel(
        #     names[ii],
        #     size=args.labelsize*1.1,
        #     labelpad=6,
        # )
            # ax_dict['data_count'].set_ylabel(
            #     r"$Count$",
            #     size=args.labelsize,
            #     labelpad=6,
            # )

        col_idx += 1
        if not col_idx % gridspec_dim['ncols']:
            row_idx, col_idx = row_idx + 1, 0

    # fig.tight_layout()
    plt.savefig(
        './figures/{}.pdf'.format(args.dataset),
        format='pdf',
        transparent=True,
        bbox_inches='tight'
        )

    if not predictive_distribution == 'Normal':
        #if regression let's plot individual contributions on the top of the
        #general predictions.
        c = torch.randperm(x.size(0))
        x, y= x[c], y[c]
        y_pred = model.predict(x.double())
        idxs = get_idxs(y_pred, y)

        shapley_loc, shapley_scale = model.individual_shapley_importance(
            x.double(), y, stats, 20, features, idxs
            )
        plot_bar(
            y_true=y[idxs],
            y_pred=1/(1 + np.exp(-y_pred[idxs])),
            column_size=2,
            shapley_loc=shapley_loc,
            shapley_scale=shapley_scale,
            top_k=5,
            deviation=args.deviation
            )
        plt.savefig(
            './figures/{}_classification.pdf'.format(args.dataset),
            format='pdf',
            transparent=True,
            bbox_inches='tight'
            )

    if x.size(-1) == 1:
        #if univariate, then let's show that our model can capture heterosc
        #noise.
        feature_size = len(shapley_importances)
        column_size = int(np.ceil(feature_size / args.row_size))
        gridspec_dim = {'nrows': 1, 'ncols': 1}
        figsize = [4, 3]

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(figure=fig, **gridspec_dim)
        gs_sub = gridspec.GridSpecFromSubplotSpec(
                    nrows=1,
                    ncols=1,
                    subplot_spec=gs[0, 0],
                    # wspace=0.7,
                    # width_ratios=[1.0],
                    # height_ratios=[1.0]
                )
        ax_dict['predictive'] = fig.add_subplot(gs_sub[0, 0])
        plot_predictive(
            ax_dict['predictive'],
            x,
            y,
            model,
            features,
            sigma=5,
            space=100
            )

        plt.savefig(
            './figures/{}.pdf'.format(args.dataset),
            format='pdf',
            transparent=True,
            bbox_inches='tight'
        )