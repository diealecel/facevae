import torch
from torch import nn

def bce_kld_loss(out, x, mu, logvar):
    bce = nn.BCEWithLogitsLoss(reduction = 'sum')(out, x)
    kld = -.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return bce + kld
