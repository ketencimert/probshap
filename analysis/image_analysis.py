# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 01:06:26 2025

@author: Mert
"""
import argparse
import os
import cv2
# import torch.nn as nn
from utils import transform

import numpy as np

import torch
import random 

# import sys
from pathlib import Path
# from datetime import datetime

from datasets import load_dataset
from utils import prepare_fold, to_np

import matplotlib.pyplot as plt
import shutil

def zip_folder(folder_path, output_path):
    """
    Zips a folder and its contents.

    Args:
        folder_path (str): The path to the folder to be zipped.
        output_path (str): The path and name of the output zip file (without extension).
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    shutil.make_archive(output_path, 'zip', folder_path)
    print(f"Folder '{folder_path}' zipped to '{output_path}.zip'")

def find_latest(pattern: str, root: Path):
    """Return the newest file under *root* whose name contains *pattern*."""
    newest_path = None
    newest_mtime = -1.0

    for path in root.rglob("*"):
        # print(path)
        # if 'fold_X' in path.name.lower():
        if path.is_file() and pattern.lower() in path.name.lower() \
            and model_name.lower() in path.name.lower() \
                and phi_net.lower() in path.name.lower():
            mtime = path.stat().st_mtime
            if mtime > newest_mtime:
                newest_mtime = mtime
                newest_path = path
    print(newest_path)
    return newest_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #device args
    parser.add_argument(
        '--device', default='cuda', type=str, help='device to train the model.'
    )
    parser.add_argument(
        '--model_id', default=0, type=int, help='model_id.'
    )
    parser.add_argument(
        '--dataset', default='mnist_normal_8', type=str
    )
    parser.add_argument(
        '--phi_net', default='masked', type=str
    )
    parser.add_argument(
        '--std', default=1, type=float
    )

    args = parser.parse_args()

    ###############################################################################
    ###############################################################################
    cv_folds = 1
    k = 1024
    dataset = args.dataset
    device = args.device
    phi_net = args.phi_net
    model_name =  f'ProbabilisticShapley{str(args.model_id)}'
    # original_size = 220 #this will be 220 was 224 before and 8 patches
    ###############################################################################
    ###############################################################################
    os.makedirs('./mnist_analysis', exist_ok=True)
    SEED = 11
    random.seed(SEED), np.random.seed(SEED), torch.manual_seed(SEED)
    path = str(find_latest(dataset, Path('./model_checkpoints')))

    # model = torch.load(path).to(device).eval()
    if 'False' in path.split('preprocess')[-1].split(')')[0][1:]:
        preprocess = False
    else:
        preprocess = True
    print(f'preprocess {preprocess}')
    data, features, dtypes, target_scale, predictive_distribution\
            = load_dataset(dataset)
    x, y = data[0]
    d_in = x.shape[-1]
    n = len(x)
    tr_size = int(n * 0.7)
    path = path.split('model_checkpoints/')[-1].replace('.pth','')
    path = path.split('model_checkpoints\\')[-1].replace('.pth','')
    folds = np.array(list(range(cv_folds)) * n)[:n]
    np.random.shuffle(folds)
    tr_dataloader, val_dataloader, te_dataloader, stats = prepare_fold(
        x, y, 10, 1, folds,
        device, torch.float32, dtypes, preprocess
        )
    original_size = data[1][2].shape[-1]
    x = torch.tensor(transform(data[1][0], stats))[:k].float()
    y = data[1][1][:k]
    images = data[1][2][:k]
    
    prior_loc, prior_scale, predictive_loc, \
        predictive_scale, predictions \
        = model.compute_parameters(
            x.to(device), 
            model.missingness_indicator(x).to(device)
            )
    
    def add_colorbar(mappable):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.pyplot as plt
        last_axes = plt.gca()
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mappable, cax=cax)
        plt.sca(last_axes)
        return cbar
    
    
    choice_list = []
    for index, l in enumerate(y):
        if l != 1:
            print(index)
            choice_list.append(index)
    sd = [0, 1, 2, 3]
    choice = np.random.choice(choice_list, 5, replace=False).tolist()
    fig, axes = plt.subplots(nrows=len(sd), ncols=5, figsize=(20,20))
    # for j, i in enumerate((-predictions).topk(5)[-1]):
    for k in range(len(sd)):
        ax = axes[k]
        for j, i in enumerate(choice):
            new_size = (original_size, original_size)
            loc =  prior_loc - 3 * prior_scale
            resized_prior_loc = cv2.resize(
                to_np(loc[i]).reshape(
                    int(loc[i].shape[-1]**0.5),int(loc[i].shape[-1]**0.5),1
                ), new_size, interpolation=cv2.INTER_CUBIC
                )

            ax[j].imshow(images[i])  # 0.0 (transparent) to 1.0 (opaque)
            im = ax[j].imshow(resized_prior_loc, cmap='jet', alpha=0.7)

            add_colorbar(im)
            ax[j].set_title(f'Prediction: {round(predictions[i].item(), 3)}')

    plt.tight_layout()
    plt.savefig(f'./mnist_analysis/negative_mnist_{path}.pdf')
    # plt.show()
    
    choice_list = []
    for index, l in enumerate(y):
        if l == 1:
            print(index)
            choice_list.append(index)
    sd = [0, 1, 2, 3]
    choice = np.random.choice(choice_list, 5, replace=False).tolist()
    fig, axes = plt.subplots(nrows=len(sd), ncols=5, figsize=(20,20))
    #for j, i in enumerate((predictions).topk(5)[-1]):
    for k in range(len(sd)):
        ax = axes[k]
        for j, i in enumerate(choice):
            new_size = (original_size, original_size)
            loc =  prior_loc + sd[k] * prior_scale
            resized_prior_loc = cv2.resize(
                to_np(loc[i]).reshape(int(loc[i].shape[-1]**0.5),int(loc[i].shape[-1]**0.5),1), new_size, interpolation=cv2.INTER_CUBIC
                )

            ax[j].imshow(images[i])  # 0.0 (transparent) to 1.0 (opaque)
            im = ax[j].imshow(resized_prior_loc, cmap='jet', alpha=0.7)
            # cax = inset_axes(ax[j], width="3%", height="100%", loc="lower left",
            #              bbox_to_anchor=(1.02, 0., 1, 1), bbox_transform=ax[j].transAxes,
            #              borderpad=1)
            add_colorbar(im)
            ax[j].set_title(f'Prediction: {round(predictions[i].item(), 3)}')
    
        # divider = make_axes_locatable(ax[j])
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # plt.colorbar(im, ax=ax[j])
    plt.tight_layout()
    plt.savefig(f'./mnist_analysis/positive_mnist_{path}.pdf')

    zip_folder('mnist_analysis', 'mnist_analysis')

# from transform import * 
# # i = 3
# autoencoder = torch.load('./transform_mnist_8_segments.pt')
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.Resize((224,224)),
#                        transforms.ToTensor(),
#                        ])),
#     batch_size=k, shuffle=False,
#     )
# autoencoder_input = next(iter(train_loader))

# choice_list = []
# for index, l in enumerate(autoencoder_input[1]):
#     if l == 2:
#         print(index)
#         choice_list.append(index)

# i = np.random.choice(choice_list, 1).item()
  
# encoded_input = autoencoder.encoder(autoencoder_input[0].cuda())
# encoded_input_to_shapley_net = torch.tensor(
#     transform(to_np(encoded_input), stats)
#     ).cuda()
# prior_loc, prior_scale, predictive_loc, \
#     predictive_scale, predictions \
#     = model.compute_loc_scale(
#         encoded_input_to_shapley_net, 
#         torch.ones_like(encoded_input_to_shapley_net)
#         )
# plt.imshow(to_np(autoencoder_input[0][i]).reshape(224,224))
# plt.show()
# loc = torch.distributions.Normal(prior_loc, prior_scale).sample()
# decoded_image = autoencoder.decoder(prior_scale)
# plt.imshow(to_np(decoded_image[i]).reshape(224,224))
# plt.show()
# import torch

# def conditional_x_given_y(mu_x, cov_x, loc_y, sigma_y):
#     """
#     Computes the conditional distribution p(x | y)
#     where y = sum(x) + epsilon, epsilon ~ N(0, sigma_y^2)

#     Args:
#         mu_x: (D,) tensor, mean of x
#         cov_x: (D, D) tensor, covariance of x
#         loc_y: scalar tensor, observed value of y
#         sigma_y: scalar tensor, stddev of noise

#     Returns:
#         cond_mu: (D,) tensor, conditional mean of x given y
#         cond_cov: (D, D) tensor, conditional covariance of x given y
#     """
#     D = mu_x.shape[0]
#     one_vec = torch.ones(D, dtype=mu_x.dtype, device=mu_x.device)

#     # Compute scalar terms
#     mu_y = one_vec @ mu_x  # scalar
#     sigma_yy = one_vec @ cov_x @ one_vec + sigma_y**2  # scalar

#     # Cross-covariance
#     sigma_xy = cov_x @ one_vec  # shape: (D,)

#     # Posterior mean
#     cond_mu = mu_x + sigma_xy * ((loc_y - mu_y) / sigma_yy)

#     # Posterior covariance
#     cond_cov = cov_x - torch.outer(sigma_xy, sigma_xy) / sigma_yy

#     return cond_mu, cond_cov

# cond_mu, cond_cov = conditional_x_given_y(
#     prior_loc[i],
#     torch.diag(prior_scale[i]).pow(2),
#     1000,
#     nn.Softplus()(model.predictive.scale)
#     )

# loc = cond_mu + cond_cov @ torch.ones(cond_mu.size()).cuda()

# resized_prior_loc = cv2.resize(
#     to_np(loc).reshape(8,8,1), new_size, interpolation=cv2.INTER_CUBIC
#     )
# plt.imshow(images[i])  # 0.0 (transparent) to 1.0 (opaque)
# plt.imshow(resized_prior_loc, cmap='jet', alpha=0.6)
# plt.colorbar() 
# plt.show()