import torch
from torch.nn import functional as nnf

def bce_kld_loss(out, x, mu, logvar):
    batch_size = x.shape[0]
    flattened_x = x.view(batch_size, -1)
    bce = nnf.binary_cross_entropy(out, flattened_x, reduction = 'sum')
    kld = -.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return bce + kld
