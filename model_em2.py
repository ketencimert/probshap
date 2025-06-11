# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 17:58:12 2022

@author: Mert
"""
from collections import defaultdict
import numpy as np

import pandas as pd

import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli

from tqdm import tqdm

from utils import inverse_transform, to_np, sample_covariance
from modules import create_masked_layers, create_feedforward_layers

# from truncated_normal import TruncatedNormal
import torchrl


class P_f_(nn.Module):
    def __init__(self, d_in, d_hid, n_layers, activation, norm, p):
        super(P_f_, self).__init__()
        # =============================================================================
        #         This is predictive net where you sum latent Shapley values to come up
        #         to y or logits, l
        # =============================================================================
        self.bias = nn.Parameter(
            torch.randn(1), requires_grad=True
        )
        self.scale = nn.Parameter(
            torch.randn(1), requires_grad=True
        )

    def forward(self, q_phi_x_loc, q_phi_x_scale):
        loc = q_phi_x_loc.sum(-1) + self.bias
        scale = torch.pow(
            q_phi_x_scale.pow(2).sum(-1) + nn.Softplus()(self.scale).pow(2),
            0.5
        )

        return loc.squeeze(-1), scale.squeeze(-1) + 1e-5


class Masked_q_phi_x(nn.Module):
    def __init__(
            self, d_in, d_hid, d_out, d_emb, n_layers, activation, norm, p,
    ):
        super(Masked_q_phi_x, self).__init__()
        # =============================================================================
        #         Masked net as described in the paper.
        # =============================================================================
        self.placeholder = nn.Parameter(
            torch.randn(d_in, d_emb), requires_grad=True
            )

        self.cont_emb_loc = nn.Sequential(
            *create_masked_layers(
                d_in=d_in, d_hid=d_hid, d_emb=d_emb, n_layers=n_layers - 2,
                activation=activation, dropout=p,
                norm=None, dtype=torch.float32
            )
        )

        self.cont_emb_shapley = nn.Sequential(
            *create_masked_layers(
                d_in=d_in, d_hid=d_hid, d_emb=d_emb, n_layers=n_layers - 2,
                activation=activation, dropout=p,
                norm=None, dtype=torch.float32
            )
        )
        self.scale_net = nn.Sequential(
            nn.Sequential(
                *create_feedforward_layers(
                    d_emb * 2, d_hid, 1, n_layers, activation, p, norm
                )
            )
        )

        self.loc_net = nn.Sequential(
            *create_feedforward_layers(
                d_emb, d_hid, d_out, n_layers, activation, p, norm
            )
        )

        self.combine_loc = nn.Linear(d_in, 1)

    def forward(self, x, m):
        # 1. Generate embeddings
        x_emb_loc = self.cont_emb_loc(x).view(x.size(0), x.size(-1), -1)
        m_ = m.repeat_interleave(
            x_emb_loc.size(-1), 1
        ).view(x.size(0), x.size(-1), -1)
        x_emb_loc = (
                x_emb_loc.masked_fill(m_ == 0, 0) + (
                self.placeholder * (1 - m_ * 1.)
        )
        )
        x_emb_loc_ = x_emb_loc.clone().detach()
        x_emb_loc = x_emb_loc.view(x.size(0), -1)
        # 2. Combine embeddings
        x_emb_loc = self.combine_loc(
            x_emb_loc.view(
                x_emb_loc.size(0), x.size(1), -1
            ).transpose(-1, -2)
        ).squeeze(-1)
        # 3. Get Shapley values
        loc = self.loc_net(x_emb_loc)
        shap_emb_ = self.cont_emb_shapley(
            loc.detach()
        ).view(x.size(0), x.size(-1), -1)
        # 4. Get Shapley uncertainty using shapley embeddings
        # and input location x. One x_d can go to many shapley values
        # (dependening on rest of x_{-d}. Hence, we use its shapley value
        # \phi_d along with x_d to get \sigma_d
        scale = nn.Softplus()(
            self.scale_net(torch.cat([shap_emb_, x_emb_loc_], -1))
        )
        # =============================================================================
        #         if a variable is marginalized, we certainly know that it will be 0.
        #         Hence, 0 location.
        #         Therefore multiply by m if you want
        # =============================================================================

        return loc.squeeze(-1), \
               scale.squeeze(-1) + 1e-5


class Vanilla_q_phi_x(nn.Module):
    def __init__(
            self, d_in, d_hid, d_out, n_layers, activation, norm, p,
            baseline
    ):
        super(Vanilla_q_phi_x, self).__init__()
        # =============================================================================
        #         Standard masked neural network.
        # =============================================================================
        self.baseline = nn.Parameter(
            torch.tensor(baseline), requires_grad=False
        )

        self.scale_net = nn.Sequential(
            nn.Sequential(
                *create_feedforward_layers(
                    3 * d_in, d_hid, d_in, n_layers, activation, p, norm
                )
            )
        )

        self.loc_net = nn.Sequential(
            *create_feedforward_layers(
                2 * d_in, d_hid, d_in, n_layers, activation, p, norm
            )
        )

    def forward(self, x, m):
        m = m * 1.
        x = x * m + (1 - m) * self.baseline.view(1, -1)
        loc = self.loc_net(torch.cat([x, m], -1))
        scale = nn.Softplus()(self.scale_net(
            torch.cat([loc.detach(), x, m], -1))
        )

        return loc.squeeze(-1), \
               scale.squeeze(-1) + 1e-5


class Model(nn.Module):
    def __init__(
            self, d_in, d_hid, d_out, d_emb,
            d_data, n_layers, activation, norm, p, beta,
            likelihood, phi_net, cont
    ):
        super(Model, self).__init__()
        
        #control variatite
        self.cont = cont
        #logit limit for numerical issues(should be inf but whatever)
        self.limit = 10.
        #beta for kl term
        self.beta = beta
        #model likelihood
        self.likelihood = likelihood
        
        self.gamma = nn.Parameter(torch.randn(1), requires_grad = True)
        
        self.p_f_ = P_f_(
            d_in, d_hid,
            n_layers, activation, norm, p,
        )

        if phi_net == 'masked':
            self.q_phi_x = Masked_q_phi_x(
                d_in, d_hid, d_in, d_emb,
                n_layers, activation, norm, p,
            )
        else:
            self.q_phi_x = Vanilla_q_phi_x(
                d_in, d_hid, d_in,
                n_layers, activation, norm, p,
                baseline=3
            )

    def predict(self, x, numpy=True):
        # =============================================================================
        #         Method to make py_x_loc. No missing values.
        # =============================================================================
        with torch.no_grad():
            # 1. Generate missing values vector of all 1s.
            x = torch.tensor(x, device=self.q_phi_x.loc_net[0].weight.device)
            m = torch.ones_like(x)
            # 2. Compute the parameters
            q_phi_x_loc, q_phi_x_scale, p_f_x_loc, \
            p_f_x_scale, py_x_loc = self.compute_parameters(x, m)
            if numpy:
                # 3. Return numpy if you want to
                py_x_loc = to_np(py_x_loc)
        return py_x_loc

    def compute_parameters(self, x, m):
        # =============================================================================
        #         Compute parameters necessary for loss computation and py_x_loc/preds.
        # =============================================================================
        # 1. Compute shapley location and scale
        # q(\phi | x)
        q_phi_x_loc, q_phi_x_scale = self.q_phi_x(x, m)
        # 2. Compute predictive distribution, in classification settings, this
        # is latent logits.
        # q(f | x) = \int p(f | \phi) q(\phi | x) d\phi
        qp_f_x_loc, qp_f_x_scale = self.p_f_(q_phi_x_loc, q_phi_x_scale)

        if self.likelihood == 'Normal':
            py_x_loc = qp_f_x_loc

        # 3. If Bernoulli feed to sigmoid function [0,1]
        elif self.likelihood == 'Bernoulli':
            py_x_loc  = 1 - Normal(
                qp_f_x_loc,
                torch.pow(
                    qp_f_x_scale.pow(2)\
                        + nn.Softplus()(self.p_f_.scale).pow(2),
                        0.5
                        )
                    ).cdf(torch.zeros_like(qp_f_x_loc))

        return q_phi_x_loc, q_phi_x_scale, qp_f_x_loc, \
               qp_f_x_scale, py_x_loc

    def missingness_indicator(self, x):
        # =============================================================================
        #         Compute missingness indicators for Shapley values i.e., m \sim p(s)
        # =============================================================================
        if self.training:

            feature_idx_init = torch.tensor(
                np.random.choice(range(x.size(-1)), x.size(0))
            ).to(x.device)
            feature_idx = feature_idx_init.unsqueeze(-1)

            permutation = torch.argsort(
                torch.rand(
                    x.size(0),
                    x.size(-1)),
                dim=-1
            ).to(x.device)

            arange = torch.arange(x.size(-1)).unsqueeze(0).repeat_interleave(
                permutation.size(0), 0
            ).to(x.device)
            pointer = arange <= torch.argmax(
                (permutation == feature_idx) * 1., -1
            ).view(-1, 1)
            p_sorted = (-permutation).topk(
                permutation.size(-1), -1, sorted=True
            )[1]
            m = torch.cat(
                [
                    torch.diag(
                        pointer[:, p_sorted[:, i]]
                    ).view(-1, 1) for i in range(
                    p_sorted.size(-1)
                )
                ], -1
            )

        else:
            m = torch.ones_like(x) == 1
        return m

    def forward(self, x, y, i):
        # =============================================================================
        #         Function where we compute all the loss functions, phi_mse stats, and
        #         make predictions
        # =============================================================================
        # 1. Generate missing values vector m \sim p(s), where m is 0
        # model won't see corresponding x values.
        original_size = x.size(0)
        m = self.missingness_indicator(x)
        x = torch.cat([x, x],0)
        m = torch.cat([m, torch.ones_like(m)],0)
        # 2. Compute distributions sufficient statistics.
        q_phi_x_loc, q_phi_x_scale, qp_f_x_loc, \
        qp_f_x_scale, py_x_loc \
            = self.compute_parameters(x, m)

        q_phi_x_loc, q_phi_x_loc_ = q_phi_x_loc.split(original_size, 0)
        q_phi_x_scale, q_phi_x_scale_ = q_phi_x_scale.split(original_size, 0)
        qp_f_x_loc, qp_f_x_loc_ = qp_f_x_loc.split(original_size, 0)
        qp_f_x_scale, qp_f_x_scale_ = qp_f_x_scale.split(original_size, 0)
        py_x_loc, py_x_loc_ = py_x_loc.split(original_size, 0)

        x = x[:original_size]
        m = m[:original_size]
        # 3. Compute an approximation to p(\phi | x):
        p_phi_x_loc1, p_phi_x_loc2, feature_idx \
            = self.p_phi_x_parameters(x, m)

        qp_f_x = Normal(qp_f_x_loc, qp_f_x_scale)

        if self.likelihood == 'Normal':
            loglikelihood = qp_f_x.log_prob(y).mean(0)

        elif self.likelihood == 'Bernoulli':
            low = torch.zeros_like(y)
            high = torch.zeros_like(y)
            low = low - (y == 0) * self.limit
            high = high + (y == 1) * self.limit
            # pf_xy = TruncatedNormal(
            #     qp_f_x_loc_, qp_f_x_scale_, a, b
            #     )
            #tanh_loc makes it more stable. so use this implementation instead.
            pf_xy = torchrl.modules.TruncatedNormal(
                loc=qp_f_x_loc_.unsqueeze(-1), 
                scale=qp_f_x_scale_.unsqueeze(-1), 
                low=low.unsqueeze(-1),
                high=high.unsqueeze(-1),
                tanh_loc=True
                )

            pf_xy_loc = pf_xy.mean.squeeze(-1)
            pf_xy_var = pf_xy.variance.squeeze(-1).clamp(min=1e-5)

            logits = pf_xy.sample().detach()

            #M-step: Evaluate the ELBO and maximize
            loglikelihood = - 1/2 * torch.log(
                2 * np.pi * qp_f_x_scale.pow(2)
                )
            loglikelihood -= (
                pf_xy_var + (pf_xy_loc - qp_f_x_loc).pow(2)
                ) / (2 * qp_f_x_scale.pow(2))
            loglikelihood += Bernoulli(
                logits=pf_xy.rsample().squeeze(-1)
                ).log_prob(y)
            # loglikelihood += pf_xy.entropy().mean(0)
            loglikelihood += pf_xy.entropy()
        # =============================================================================
        #         This is where we compute an unbiased estimate to the kl-divergence
        #         term.
        # =============================================================================
        q_phi_x_loc = q_phi_x_loc.gather(
            1, feature_idx.long().unsqueeze(-1)
        ).squeeze(-1)
        beta = self.beta
        # beta = torch.ones_like(beta)
        q_phi_x_scale = q_phi_x_scale.gather(
            1, feature_idx.long().unsqueeze(-1)
        ).squeeze(-1)
        p_phi_x_scale = q_phi_x_scale

        q_phi_x = Normal(
            q_phi_x_loc,
            q_phi_x_scale
        )
        p_phi_x = Normal(
            (p_phi_x_loc1 + p_phi_x_loc2) / 2,
            p_phi_x_scale
        )

        kld = torch.distributions.kl.kl_divergence(q_phi_x, p_phi_x)
        elbo = loglikelihood - beta * kld
        elbo = elbo.mean(0)
        
        delta_red = p_phi_x_loc2 - q_phi_x_loc
        delta_blue = p_phi_x_loc1 - q_phi_x_loc
        proxy_kld = torch.abs(delta_blue * delta_red.detach()) / (
                1e-5 + 2 * p_phi_x_scale.pow(2)
                )
        #add shapley regularization
        loss = loglikelihood - beta * proxy_kld

        #if bernoulli use control variates to control for variance?
        if self.likelihood == 'Bernoulli' and self.cont:
            h = pf_xy.log_prob(logits)
            a = sample_covariance(loss, h) / sample_covariance(h,h)
            loss -= a * h.mean(0)
        loss = loss.mean(0)
        
        return elbo, loss, proxy_kld, py_x_loc, q_phi_x_loc.mean()

    def p_phi_x_parameters(
            self,
            x,
            m,
            sample_size=1,
    ):
        # =============================================================================
        #         This is for calculation of Shapley constraint KLD as explained in the
        #         paper.
        # =============================================================================
        sample_size *= 2
        orig_size = x.size(0)
        feature_idx_init = torch.tensor(
            np.random.choice(range(x.size(-1)), x.size(0))
        ).to(x.device)
        feature_idx = feature_idx_init.repeat_interleave(
            sample_size, 0
        ).unsqueeze(-1)
        m_ = m.repeat_interleave(
            sample_size, 0
        )

        permutation = torch.argsort(
            torch.rand(
                sample_size * x.size(0),
                x.size(-1)),
            dim=-1
        ).to(x.device)

        arange = torch.arange(x.size(-1)).unsqueeze(0).repeat_interleave(
            permutation.size(0), 0
        ).to(x.device)
        pointer = arange <= torch.argmax(
            (permutation == feature_idx) * 1., -1
        ).view(-1, 1)
        p_sorted = (-permutation).topk(
            permutation.size(-1), -1, sorted=True
        )[1]
        m1 = torch.cat(
            [
                torch.diag(
                    pointer[:, p_sorted[:, i]]
                ).view(-1, 1) for i in range(
                p_sorted.size(-1)
            )
            ], -1
        )
        m2 = m1.masked_fill(arange == feature_idx, False)
        m_repeat = torch.cat([m1 * m_, m2 * m_], 0)
        x_repeat = torch.cat(
            [
                x.repeat_interleave(sample_size, 0),
                x.repeat_interleave(sample_size, 0)
            ],
            0
        )
        q_phi_x_loc, q_phi_x_scale, qp_f_x_loc, \
        qp_f_x_scale, py_x_loc = self.compute_parameters(
            x_repeat, m_repeat
        )

        p1, p2 = qp_f_x_loc.split(qp_f_x_loc.size(0) // 2, 0)

        shapley_loc = (p1 - p2).view(orig_size, -1)
        shapley_loc1, shapley_loc2 = shapley_loc.split(
            shapley_loc.size(-1) // 2, -1
        )
        shapley_loc1 = shapley_loc1.mean(-1)
        shapley_loc2 = shapley_loc2.mean(-1)

        return shapley_loc1, shapley_loc2, feature_idx_init

    ###############################################################################
    # =============================================================================
    #     These are for plotting - interpretability. After here there is nothing
    #     that relates to model but only using model to plot Shapley values.
    # =============================================================================
    ###############################################################################
    def feature_importances(
            self, x, stats, features,
            dtypes, interactions=False, estimate='monte_carlo',
    ):
        batch_size = 128
        with torch.no_grad():
            shapley_importances = defaultdict(list)
            for i in tqdm(range(x.size(-1))):
                for x_b in x.split(batch_size, 0):
                    importance_stats = self.first_order_shapley_importance(
                        x_b,
                        i,
                        estimate=estimate
                    )
                    importance_stats[:, 0] = inverse_transform(
                        importance_stats[:, 0], stats[:, i]
                    )
                    if dtypes[i] != 'uint8':
                        shapley_importances[
                            (features[i], dtypes[i])
                        ].append(importance_stats)
                    else:
                        importance_stats = importance_stats[
                                           abs(importance_stats[:, 0] - 1) < 1e-4, :
                                           ]
                        feature = features[i].split('_')
                        f = np.asarray(
                            [feature[-1]] * importance_stats.shape[0]
                        ).reshape(-1, 1)
                        importance_stats = np.concatenate(
                            [f, importance_stats], -1
                        )
                        importance_stats = np.delete(
                            importance_stats, 1, axis=1
                        )
                        shapley_importances[
                            (feature[0], dtypes[i])
                        ].append(importance_stats)

        for key in shapley_importances.keys():
            shapley_importances[key] = np.concatenate(
                shapley_importances[key]
            )
        return shapley_importances

    def first_order_shapley_importance(
            self, x, feature_idx, sample_size=30, order=True,
            estimate='MONTECARLO'
    ):
        with torch.no_grad():
            if order:
                x = x[x[:, feature_idx].sort()[1]]

            if estimate == 'MONTECARLO':
                idx = torch.stack(
                    [
                        torch.randperm(x.size(-1)) for _ in range(
                        sample_size * x.size(0)
                    )
                    ]
                ).to(x.device)
                pointer = torch.arange(x.size(-1)).unsqueeze(0).repeat_interleave(
                    idx.size(0), 0
                ).to(x.device)
                pointer = pointer <= torch.argmax(
                    (idx == feature_idx) * 1., -1
                ).view(-1, 1)
                p_sorted = (-idx).topk(idx.size(-1), -1, sorted=True)[1]
                missing = torch.cat(
                    [
                        torch.diag(
                            pointer[:, p_sorted[:, i]]
                        ).view(-1, 1) for i in range(
                        p_sorted.size(-1)
                    )
                    ], -1
                )
                x_repeat = x.repeat_interleave(sample_size, 0)
                q_phi_x_loc1, q_phi_x_scale1, qp_f_x_loc1, \
                qp_f_x_scale1, py_x_loc1 = self.compute_parameters(
                    x_repeat, missing
                )

                missing[:, feature_idx] = False
                q_phi_x_loc2, q_phi_x_scale2, qp_f_x_loc2, \
                qp_f_x_scale2, py_x_loc2 = self.compute_parameters(
                    x_repeat, missing
                )

                f_k_mean = torch.stack(
                    [
                        l.mean() for l in (
                            py_x_loc1 - py_x_loc2
                    ).split(sample_size, -1)
                    ],
                    -1
                ).unsqueeze(-1)

                # f_k_scale = torch.tensor(
                #     [torch.nan] * f_k_mean.size(0),
                #     device=f_k_mean.device
                #     ).unsqueeze(-1)
                # this is wrong as the variance will not be equal to the variance in coalitions
                f_k_scale = torch.stack(
                    [
                        l.std() for l in (
                            py_x_loc1 - py_x_loc2
                    ).split(sample_size, -1)
                    ],
                    -1
                ).unsqueeze(-1)

            elif estimate == 'FEEDFORWARD':

                q_phi_x_loc, q_phi_x_scale, qp_f_x_loc, \
                qp_f_x_scale, py_x_loc = self.compute_parameters(
                    x, torch.ones_like(x)
                )

                f_k_mean = q_phi_x_loc[:, feature_idx].unsqueeze(-1)
                f_k_scale = q_phi_x_scale[:, feature_idx].unsqueeze(-1)

        feature_value = x[:, feature_idx].unsqueeze(-1)
        importance_stats = torch.cat(
            [feature_value, f_k_mean, f_k_scale], -1
        )

        return to_np(importance_stats)

    def individual_shapley_importance(
            self, x, y, stats, sample_size, features, idxs
    ):
        expected_value = y.mean()
        shapley_importances_loc = defaultdict(dict)
        shapley_importances_scale = defaultdict(dict)
        with torch.no_grad():
            x_ = x[idxs, :]
            for i in range(x.size(-1)):
                importance_stats = self.first_order_shapley_importance(
                    x_,
                    i,
                    sample_size=sample_size,
                    order=False
                )

                importance_stats[:, 0] = inverse_transform(
                    importance_stats[:, 0], stats[:, i]
                )

                for j, importance_stat in enumerate(importance_stats):
                    shapley_importances_loc[
                        idxs[j]
                    ][
                        (features[i], importance_stat[0])
                    ] = importance_stat[1] + expected_value / x.size(
                        -1
                    )
                    shapley_importances_scale[idxs[j]][
                        (features[i], importance_stat[0])
                    ] = importance_stat[2]
            shapley_importances_loc = pd.DataFrame(shapley_importances_loc)
            shapley_importances_scale = pd.DataFrame(shapley_importances_scale)
            return shapley_importances_loc, shapley_importances_scale

    def shapley_values(self, x, estimate='FEEDFORWARD', sample_size=10):

        if estimate == 'FEEDFORWARD':
            shapley_values = to_np(self.q_phi_x(x, torch.ones_like(x))[0])
        elif estimate == 'MONTECARLO':
            shapley_values = []
            for i in range(x.size(-1)):
                shapley_values.append(
                    self.first_order_shapley_importance(
                        x=x,
                        feature_idx=i,
                        sample_size=sample_size,
                        order=False,
                        estimate='MONTECARLO'
                    )[:, 1].reshape(-1, 1)
                )
            shapley_values = np.concatenate(shapley_values, -1)
        return shapley_values