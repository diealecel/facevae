import torch
from torch import nn

from neural.layers import Flatten, Reshape

class BaselineVAE(nn.Module):
    def __init__(self):
        super(BaselineVAE, self).__init__()

        self.encode_nn_mu = nn.Sequential(
            Flatten(),
            nn.Linear(30000, 30)
        )

        self.encode_nn_logvar = nn.Sequential(
            Flatten(),
            nn.Linear(30000, 30)
        )

        self.decode_nn = nn.Sequential(
            nn.Linear(30, 30000),
            Reshape((3, 100, 100))
        )

    def encode(self, x):
        mu = self.encode_nn_mu(x)
        logvar = self.encode_nn_logvar(x)

        return mu, logvar


    def reparametrize(self, mu, logvar):
        std = torch.exp(.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std


    def decode(self, z):
        return self.decode_nn(z)


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        out = self.decode(z)

        return out, mu, logvar
