# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 13:44:10 2023

@author: Mert
"""
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter1d
from scipy import interpolate

from utils import inverse_transform

import pandas as pd
import seaborn as sns

import torch
from utils import to_np

plt.style.use('seaborn-paper')

sns.set_palette("Dark2")
WEIGHTS_COLOR = '#191919' #bc9293
# WEIGHTS_VALUE_COLOR = '#2D4263'  # '#fdbf6f' # '#ff7f00'
DENSITY_COLOR = '#F05454'  #65a7cc

# WEIGHTS_COLOR = '#191919' #bc9293
WEIGHTS_VALUE_COLOR = '#2D4263'  # '#fdbf6f' # '#ff7f00'
# DENSITY_COLOR = '#65a7cc'  #65a7cc
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

cdict1 = {
    'red': ((0.0, 0.11764705882352941, 0.11764705882352941),
            (1.0, 0.9607843137254902, 0.9607843137254902)),

    'green': ((0.0, 0.5333333333333333, 0.5333333333333333),
              (1.0, 0.15294117647058825, 0.15294117647058825)),

    'blue': ((0.0, 0.8980392156862745, 0.8980392156862745),
             (1.0, 0.3411764705882353, 0.3411764705882353)),

    'alpha': ((0.0, 1, 1),
              (0.5, 1, 1),
              (1.0, 1, 1))
    }


def interpolation(x, y, scale, gf_sigma, deg=1, min_knot_size=15):
    while min_knot_size>0:
        try:
            knot_size = min(min_knot_size,len(np.unique(x)) // 4)
            knot_positions = np.linspace(
                np.quantile(x,0.05)+1e-5,
                np.quantile(x,0.95)-1e-5,
                knot_size
                )
            interpolator_y = interpolate.LSQUnivariateSpline(
                x, y, knot_positions, k=deg
                )
            interpolator_scale = interpolate.LSQUnivariateSpline(
                x, scale, knot_positions, k=deg
                )
            x_ = np.arange(x.min(), x.max(), 0.1)

            y_ = gaussian_filter1d(interpolator_y(x_), gf_sigma)
            scale_ = gaussian_filter1d(interpolator_scale(x_), gf_sigma)
            x_ = gaussian_filter1d(x_, gf_sigma)
            return x_, y_, scale_
        except:
            min_knot_size -= 1
    return [None] * 3

def plot_2d(feature_name, ax_dict, feature_dict, fs_, labelsize=15):

    ax1, ax2 = ax_dict['importance'], ax_dict['real']

    x = feature_dict[:,:2]
    y = feature_dict[:,-1]
    c1 = ax1.tricontourf(
        x[:,0],
        x[:,1],
        y
        )
    plt.colorbar(c1)
    ax1.set_title('$Estimated$', size=labelsize, fontweight="bold")
    ax1.tick_params(axis='both', labelsize=labelsize)
    ax1.set_xlabel('${}$'.format(feature_name[0]),
                  size=labelsize,
                  fontweight="bold"
                  )
    ax1.set_ylabel('${}$'.format(feature_name[1]),
                  size=labelsize,
                  fontweight="bold"
                  )
    ax1.xaxis.set_tick_params(rotation=45)

    n = int(fs_.shape[0]**0.5)
    c2 = ax2.contourf(
        fs_[:,0].reshape(n, n),
        fs_[:,1].reshape(n, n),
        fs_[:,2].reshape(n, n),
        )
    plt.colorbar(c2)
    ax2.set_title('$True$', size=labelsize, fontweight="bold")
    ax2.tick_params(axis='both', labelsize=labelsize)
    ax2.set_xlabel('${}$'.format(feature_name[0]),
                  size=labelsize,
                  fontweight="bold"
                  )
    # ax2.set_ylabel('${}$'.format(feature_name[1]),
    #               size=labelsize,
    #               fontweight="bold"
    #               )
    ax2.xaxis.set_tick_params(rotation=45)
    plt.tight_layout()
    return ax_dict

def plot_1d(
        feature_name, ax_dict, feature_dict, deviation, gf_sigma=10,
        fs_=None, bin_size=50, density=True, labelsize=15,
        deg=1, min_knot_size=15, subsample_ratio=0.1
        ):

    ax1, ax2 = ax_dict['importance'], ax_dict['data_count']
    histogram = len(ax_dict.keys()) == 2
    feature_dict = feature_dict[np.lexsort(np.fliplr(feature_dict).T)]
    x = feature_dict[:,0]
    y = feature_dict[:,1].astype(float)
    scale = deviation * feature_dict[:,2].astype(float)

    x_, y_ , scale_ = interpolation(x, y, scale, gf_sigma, deg, min_knot_size)

    if x_ is not None:

        ax1.plot(
            x_,
            y_,
            color='black',
            alpha=1,
            label='Estimated'
            )
        if scale_ is not np.nan:
            ax1.fill_between(
                x_,
                y_ - scale_,
                y_ + scale_,
                color=WEIGHTS_COLOR,
                alpha=0.2
                )
        idx_sample = np.random.choice(
            x.shape[0], int(x.shape[0] * subsample_ratio), False
            )
        ax1.scatter(
            x[idx_sample],
            y[idx_sample],
            s=15,
            alpha=0.5,
            )
        if fs_ is not None:
            fs_x, fs_y = zip(*sorted(zip(fs_[:,1], fs_[:,0])))
            fs_y = gaussian_filter1d(
                fs_y,
                sigma=gf_sigma,
                )
            ax1.plot(
                fs_x,
                fs_y,
                linestyle ='--',
                color='red',
                label='True'
                )

        ax1.set_title(
            r'${}$'.format(feature_name), size=labelsize, fontweight="bold"
            )
        ax1.tick_params(axis='both', labelsize=labelsize)

        if histogram:
            ax1.get_xaxis().set_visible(False)
            ax2.hist(
                x=x,
                bins=bin_size,
                density=density,
                edgecolor='black',
                facecolor=DENSITY_COLOR,
                )
            ax2.tick_params(axis='both', labelsize=labelsize)

        else:
            ax1.get_xaxis().set_visible(True)

    else:

        data = pd.DataFrame([x, y]).T.groupby(0).agg(
            {0:'count', 1: 'mean'}
            ).sort_values(1)
        ax1.scatter(
            x=list(data.index), y=data[1], s=100, color='black', marker='x'
            )
        ax1.errorbar(
            x=x,
            y=y,
            yerr=scale,
            elinewidth=2.0,
            capsize=3.0,
            marker='.',
            linestyle='',
            alpha=0.65,
            markersize=10,
        )

        ax1.set_title(
            r'${}$'.format(feature_name), size=labelsize, fontweight="bold"
            )
        plt.xticks(rotation=45, ha='right')
        ax1.tick_params(axis='both', labelsize=labelsize)

        if histogram:

            ax1.get_xaxis().set_visible(False)
            ax2.bar(
                x=list(data.index),
                height=data[0],
                edgecolor='black',
                facecolor=DENSITY_COLOR,
            )
            ax2.tick_params(axis='both', labelsize=labelsize)

    return ax_dict

def plot_bar(
        y_true, y_pred, column_size, shapley_loc,
        shapley_scale, top_k=5, deviation=1
        ):

    row_size = int(np.ceil(shapley_loc.shape[-1] / column_size))
    figsize = [4 * column_size , 4 * row_size]
    fig = plt.figure(figsize=figsize)
    fig, ax = plt.subplots(nrows=row_size, ncols=column_size)

    for i in range(column_size):
        for j in range(row_size):
            k = i*2**i + j
            key = shapley_loc.keys()[k]
            shapley_loc_ = shapley_loc[key].dropna().reset_index(drop=False)
            shapley_loc_ = shapley_loc_.sort_values(key, ascending=False)
            shapley_loc_ = pd.concat(
                [shapley_loc_[:top_k], shapley_loc_[-top_k:]], 0
                )
            shapley_scale_ = shapley_scale[key].dropna().reset_index(drop=False)
            shapley =  pd.merge(
                shapley_loc_, shapley_scale_, on=["level_0", "level_1"]
                )

            features = shapley['level_0'].values
            feature_values = shapley['level_1'].values

            x_ = [
                f + ' ({})'.format(round(fv,3)) for (f, fv) in zip(
                    features,
                    feature_values
                    )
                ]
            y_ = shapley[str(key) + '_x'].values
            err_ = shapley[str(key) + '_y'].values

            pal = sns.color_palette("flare", len(feature_values))
            # rank = feature_values.argsort().argsort()
            ax[i][j].barh(
                y=np.flip(x_),
                width=np.flip(y_),
                xerr=np.asarray([[0,x] for x in np.flip(deviation*err_)]).T,
                color=np.array(pal[::-1]),
                )
            ax[i][j].set_title('True: {} Pred: {}'.format(
                y_true[k], round(y_pred[k].item(),3)))
            plt.tight_layout()

def plot_predictive(
        ax, x, y, model, features, deviation=3, sigma=10, space=1000
        ):
    labelsize = 15
    x_ = torch.linspace(
        x.min(), x.max(), space
        ).to(x.device).view(-1,1)

    y_pred, predictive_scale = model.predict(x_)
    y_pred = to_np(y_pred)
    predictive_scale = to_np(predictive_scale)

    y_pred = gaussian_filter1d(
        y_pred,
        sigma=sigma
        )
    scale = gaussian_filter1d(
        predictive_scale,
        sigma=sigma
        )
    x_ = to_np(x_).reshape(-1)
    ax.plot(x_, y_pred, color='black', linestyle='--', label='Predictions')
    ax.fill_between(
        x_,
        y_pred - deviation*scale,
        y_pred + deviation*scale,
        alpha=0.1,
        color=WEIGHTS_COLOR,
        label='Credible Intervals'
        )
    ax.scatter(
        to_np(x), y, alpha=0.05, marker='x', c='red', label='Observed Data'
        )
    ax.legend()
    ax.set_xlabel('${}$'.format(features[0]),
                  size=labelsize,
                  fontweight="bold"
                  )
    ax.tick_params(axis='both', labelsize=labelsize)
    # ax.set_ylabel('${}$'.format(''),
    #               size=labelsize,
    #               fontweight="bold"
    #               )

    plt.tight_layout()
    return ax