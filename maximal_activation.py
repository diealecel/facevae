import torch
from argparse import ArgumentParser
from os.path import join
from neural.utils import get_data_loader_with_paths

import numpy as np

import matplotlib.pyplot as plt

SAVED_MODEL_PATH = 'models/saved'
TRAIN_LOC = 'dataset/10k_500_split/train'
MODEL_SAVE_EXT = '.model'

BATCH_SZ = 64

def get_model(model):
    model_path = join(SAVED_MODEL_PATH, model + MODEL_SAVE_EXT)
    return torch.load(model_path)


def get_latents(model, train_loader, device):
    all_latents = None
    with torch.no_grad():
        for batch_idx, (data, _, _) in enumerate(train_loader):
            data = data.to(device)

            mu, logvar = model.encode(data)
            latent = model.reparametrize(mu, logvar).cpu().numpy()
            all_latents = latent if all_latents is None else np.concatenate((all_latents, latent))

    return all_latents


def get_latent_to_img_path(train_loader, all_latents):
    latent_to_img_path = {}
    with torch.no_grad():
        for batch_idx, (data, _, img_path) in enumerate(train_loader):
            num_ex = data.shape[0]
            start_idx = batch_idx * BATCH_SZ
            latents = all_latents[start_idx:start_idx + num_ex, :]
            latent_to_img_path.update(zip(img_path, np.split(latents, num_ex)))

    return latent_to_img_path


def main(args, device):
    model = get_model(args.model)
    train_loader = get_data_loader_with_paths(TRAIN_LOC, BATCH_SZ, shuffle = False)
    all_latents = get_latents(model, train_loader, device)

    print(all_latents.max())
    print(all_latents.min())
    print(all_latents.mean())
    print(np.median(all_latents))
    print(all_latents.var() ** .5)
    np.save('L', all_latents)

    latent_to_img_path = get_latent_to_img_path(train_loader, all_latents)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--no-cuda', action = 'store_true',
                        help = 'disables CUDA training')
    parser.add_argument('--model', type = str, default = None,
                        help = 'filename to save model at end of training')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device('cuda' if args.cuda else 'cpu')
    main(args, device)
