# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 01:06:26 2025

@author: Mert
"""
import cv2
import torch.nn as nn
from utils import transform
k = 512
self = model.eval()
x, y = data[1][0], data[1][1]
x = transform(x, stats)
x = torch.tensor(x)[:k]
y = y[:k]
images = data[1][2][:k]
x = x.cuda()
m = self.missingness_indicator(x)
prior_loc, prior_scale, predictive_loc, \
    predictive_scale, predictions \
    = self.compute_parameters(x, m)
    
# min_, max_ = prior_loc.min(0)[0], prior_loc.max(0)[0]
# prior_loc = (prior_loc - min_) / (max_ - min_)
# prior_loc = prior_loc * 255.
choice_list = []
for index, l in enumerate(y):
    if l == 1:
        print(index)
        choice_list.append(index)
        
i = np.random.choice(choice_list, 1).item()
loc = torch.distributions.Normal(prior_loc, prior_scale).sample()
print(predictions[i])
new_size = (224, 224)
loc =  prior_loc + 2 * prior_scale

resized_prior_loc = cv2.resize(
    to_np(loc[i]).reshape(8,8,1), new_size, interpolation=cv2.INTER_CUBIC
    )

plt.imshow(images[i])  # 0.0 (transparent) to 1.0 (opaque)
plt.imshow(resized_prior_loc, cmap='jet', alpha=0.6)
plt.colorbar() 
plt.show()

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
#     = self.compute_loc_scale(
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
#     nn.Softplus()(self.predictive.scale)
#     )

# loc = cond_mu + cond_cov @ torch.ones(cond_mu.size()).cuda()

# resized_prior_loc = cv2.resize(
#     to_np(loc).reshape(8,8,1), new_size, interpolation=cv2.INTER_CUBIC
#     )
# plt.imshow(images[i])  # 0.0 (transparent) to 1.0 (opaque)
# plt.imshow(resized_prior_loc, cmap='jet', alpha=0.6)
# plt.colorbar() 
# plt.show()