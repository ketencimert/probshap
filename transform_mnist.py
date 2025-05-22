import argparse
from copy import deepcopy
import os

import torch
import torch.nn as nn
from tqdm import tqdm

import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import random_split

import matplotlib.pyplot as plt
import numpy as np

from modules import create_feedforward_layers
from utils import to_np

# =============================================================================
# This script is to transform vision data to tabular format. In particular, we
# divide an image into patches and embed each patch to a scalar value. Later,
# given all the values (in this encoded embedding) a decoder reconstructs the
# original image given a loss function. While each patch encodes a distinct
# region of the original image, the decoder reconstructs the image using all
# these patches at the same time via a feedforward neural network.
# =============================================================================

class Decoder(nn.Module):
    def __init__(self,
                 d_in=224,
                 d_hid=300,
                 n_layers=3,
                 activation='elu',
                 norm=None,
                 p=0,
                 d_patch: int = 16
                 ):
        super().__init__()

        self.decoder = nn.Sequential(
            *create_feedforward_layers(
                int((d_in // d_patch) ** 2), d_hid, d_in * d_in,
                n_layers, activation, p, norm
            )
        )

    def forward(self, x_enc):
        return self.decoder(x_enc)


class Encoder(nn.Module):
    def __init__(self,
                 d_in=224,
                 d_hid=100,
                 d_emb=50,
                 d_channels=1,
                 n_layers=3,
                 activation='elu',
                 norm=None,
                 p=0,
                 d_patch: int = 16
                 ):
        super().__init__()

        self.embedding_dim = d_emb
        self.in_channels = d_channels
        self.number_of_patches = int(
            np.ceil(
                d_in ** 2 // d_patch ** 2
            )
        )

        self.patch_size = d_patch

        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.embedding_dim * 2,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                padding=0
            ),
            nn.Flatten(
                start_dim=2,
                end_dim=3
            )
        )
        self.embedding = nn.Sequential(
            nn.Conv1d(
                in_channels=self.number_of_patches,
                out_channels=self.number_of_patches,
                kernel_size=3,
                padding=1,
                groups=self.number_of_patches
            ),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(
                in_channels=self.number_of_patches,
                out_channels=self.number_of_patches,
                kernel_size=3,
                padding=1,
                groups=self.number_of_patches
            ),
            nn.ELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

        self.position_embedding = nn.Parameter(
            torch.randn(self.number_of_patches, self.embedding_dim)
        )
        self.combine_loc = nn.Sequential(
            *[
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.ELU(),
                nn.Linear(self.embedding_dim, 1)
            ]
        )

    def forward(self, x):
        # Perform the forward pass
        x_enc = self.patcher(x).permute(0, 2, 1)
        x_enc = self.embedding(x_enc)
        x_enc = x_enc + self.position_embedding.unsqueeze(0)

        x_enc = self.combine_loc(x_enc)

        return x_enc.squeeze(-1)


class Model(nn.Module):
    def __init__(self,
                 d_in=224,
                 d_hid=100,
                 d_emb=50,
                 d_channels=3,
                 n_layers=3,
                 activation='elu',
                 norm=None,
                 p=0,
                 d_segment=14  # number of segments
                 # d_patch:int=16
                 ):
        super().__init__()

        d_patch = (d_in // d_segment)
        self.encoder = Encoder(d_in=d_in, d_patch=d_patch)
        self.decoder = Decoder(d_in=d_in, d_patch=d_patch)

    def forward(self, x):
        x_enc = self.encoder(x)
        x_dec = self.decoder(x_enc)
        return x_dec


def train(args, model, train_loader, optimizer, epoch):
    model.train()
    loss_list = []
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(args.device)

        optimizer.zero_grad()
        x_dec = model(data)
        if args.loss == 'cb':
            loss = - torch.distributions.ContinuousBernoulli(
                logits=x_dec
            ).log_prob(data.reshape(data.size(0), -1)).sum(-1).mean(0)
        elif args.loss == 'normal':
            loss = (x_dec - data.view(data.size(0), -1)).pow(2).sum(-1).mean()
            # maybe you should not put uniform standard deviation
        loss.backward()

        optimizer.step()
        loss_list.append(loss.item())
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), np.mean(loss_list)))
    return model


def test(args, model, best_model, test_loader, best_score):
    model.eval()
    loss_list = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(args.device)

            x_dec = model(data)
            if args.loss == 'cb':
                loss = - torch.distributions.ContinuousBernoulli(
                    logits=x_dec
                ).log_prob(data.reshape(data.size(0), -1)).sum(-1).mean(0)
            elif args.loss == 'normal':
                loss = (
                        x_dec - data.view(data.size(0), -1)
                ).pow(2).sum(-1).mean()
            if args.plot:
                if args.loss == 'cb':
                    plt.imshow(
                        nn.Sigmoid()(x_dec)[10].reshape(d_in, d_in).detach().cpu()
                    )
                else:
                    plt.imshow(
                        x_dec[10].reshape(d_in, d_in).detach().cpu()
                    )
                plt.show()

            loss_list.append(loss.item())

    print(
        '\nTest set: Loss: {:.4f} \n'.format(
            np.mean(loss_list)
        )
    )
    current_score = np.mean(loss_list)
    print(f'The best score is {best_score} and current score is {current_score}.')
    if current_score <= best_score:
        print('Caching best model.')
        best_model = deepcopy(model)
        best_score = current_score

    return best_model, best_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_interval', default=10,
                        help='log per epoch.'
                        )
    parser.add_argument('--save_model', default=True,
                        help='learning rate.'
                        )
    parser.add_argument('--device', default='cuda', type=str,
                        help='device to train the model.'
                        )
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate.'
                        )
    parser.add_argument('--loss', default='normal', type=str,
                        help='loss for decoder.'
                        )
    parser.add_argument('--epochs', default=int(200), type=int,
                        help='number of training epochs.'
                        )
    parser.add_argument('--batch_size', default=1024, type=int,
                        help='batch size.'
                        )
    parser.add_argument('--d_in', default=300, type=int,
                        help='resize images (220)'
                        )
    parser.add_argument('--d_segment', default=15, type=int,
                        help='new size will be d_segment ** 2 (10)'
                        )
    # for now, fixed -- it is only mnist
    parser.add_argument('--dataset', default='lung', type=str)
    # parser.add_argument('--plot', default=True)
    parser.add_argument('--plot', default='store_true')
    args = parser.parse_args()
    # args.plot = True
    kwargs = {
        'num_workers': 1, 'pin_memory': False
    } if 'cuda' in args.device else {}

    # Let's fix these to make sure that they are thesame accross different
    # datasets.
    d_in = args.d_in  # make it stay as 24 - before 224
    d_segment = args.d_segment # make this 8 - 8 pieces, each describe 3 by 3 pixels before 14
    # Augment the data to better generalize
    train_transform = transforms.Compose([
        # transforms.CenterCrop(26),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((d_in, d_in)),
        transforms.ColorJitter(
            brightness=0.05,
            contrast=0.05,
            saturation=0.05,
            hue=0.05
        ),
        transforms.RandomRotation(10),
        transforms.RandomAffine(5),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomAffine(
            degrees=20,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.RandomInvert(),
        transforms.RandomAutocontrast(),
        transforms.RandomEqualize(),
        transforms.AugMix(),
        transforms.ToTensor(),
        ])
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((d_in, d_in)),
        transforms.ToTensor()
        ])
    if args.dataset == 'mnist':
            train_dataset = datasets.MNIST(
                '../data',
                train=True,
                download=True,
                transform=train_transform
                )
            test_dataset = datasets.MNIST(
                '../data',
                train=False,
                transform=test_transform
                )

    elif args.dataset == 'lung':
        full_dataset = datasets.ImageFolder(
            root="./data/lung/",
            transform=train_transform
            )
        train_len = int(0.8 * len(full_dataset))
        test_len = len(full_dataset) - train_len
        generator = torch.Generator().manual_seed(42)
        train_dataset, test_dataset = random_split(
            full_dataset,
            [train_len, test_len],
            generator=generator
            )
        test_dataset = deepcopy(test_dataset)

    train_dataset.dataset.transform = train_transform
    train_dataset.dataset.transforms = train_transform
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
        )

    test_dataset.dataset.transform = test_transform
    test_dataset.dataset.transforms = test_transform
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs
        )

    if os.path.exists(
            f'./transform_{args.dataset}_{d_segment}_segments_{args.loss}.pt'
            ):
        print('File found. Loading and generating results...')
        with torch.no_grad():
            kwargs['pin_memory'] = False
            best_model = torch.load(
                f'./transform_{args.dataset}_{d_segment}_segments_{args.loss}.pt'
            )
            train_dataset.dataset.transform = test_transform
            train_dataset.dataset.transforms = test_transform
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                **kwargs
                )
            split = ['train', 'test']
            for i, loader in enumerate([train_loader, test_loader]):

                images = []
                labels = []
                encoded_dataset = []

                try:
                    for batch_idx, (input_data, label) in tqdm(
                            enumerate(loader),
                            total=len(loader)
                    ):
                        images.append(to_np(input_data))
                        labels.append(to_np(label))
                        encoded_dataset.append(
                            to_np(best_model.encoder(input_data.to(args.device)))
                        )

                    np.save(
                        f'./{split[i]}_{args.dataset}_{d_segment}_segments_{args.loss}_images',
                        np.squeeze(
                            np.concatenate(images), 1)
                    )
                    np.save(
                        f'./{split[i]}_{args.dataset}_{d_segment}_segments_{args.loss}_labels',
                        np.concatenate(labels)
                    )
                    np.save(
                        f'./{split[i]}_{args.dataset}_{d_segment}_segments_{args.loss}_inputs',
                        np.concatenate(encoded_dataset)
                    )

                except:

                    np.save(
                        f'./{split[i]}_{args.dataset}_{d_segment}_segments_{args.loss}_images',
                        np.squeeze(
                            np.concatenate(images), 1)
                    )
                    np.save(
                        f'./{split[i]}_{args.dataset}_{d_segment}_segments_{args.loss}_labels',
                        np.concatenate(labels)
                    )
                    np.save(
                        f'./{split[i]}_{args.dataset}_{d_segment}_segments_{args.loss}_inputs',
                        np.concatenate(encoded_dataset)
                    )
    else:
        print('Training model.')
        print(
            'Once training complete run this script again to generate data.'
        )
        try:
            model = Model(d_in=d_in, d_segment=d_segment).to(args.device)
            best_model = None
            best_score = np.inf
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

            for epoch in range(1, args.epochs + 1):
                model = train(args, model, train_loader, optimizer, epoch)
                best_model, best_score = test(
                    args, model, best_model, test_loader, best_score
                )
            print('Training complete. Saving the model...')

        except KeyboardInterrupt:
            print('Keyboard interrupt. Saving the best model...')
            if best_model is None:
                best_model = model

        torch.save(
            best_model,
            f"./transform_{args.dataset}_{d_segment}_segments_{args.loss}.pt"
        )
