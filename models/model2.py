import torch
from torch import nn

from neural.layers import Flatten, Reshape

class VAE2(nn.Module):
    def __init__(self):
        super(VAE2, self).__init__()

        # ENCODER
        self.e_conv1 = nn.Conv2d(3, 32, 16, 10)
        self.e_bn1 = nn.BatchNorm2d(32)
        self.e_relu1 = nn.ReLU()
        self.e_flatten = Flatten()
        self.mu_fc = nn.Linear(2592, 1000)
        self.logvar_fc = nn.Linear(2592, 1000)

        
        # DECODER
        self.d_lin = nn.Linear(1000, 2592)
        self.d_reshape = Reshape((32, 9, 9))
        self.d_relu1 = nn.ReLU()
        self.convt1 = nn.ConvTranspose2d(32, 16, kernel_size = 3, stride = 3)
        self.d_relu2 = nn.ReLU()
        self.d_bn1 = nn.BatchNorm2d(16)
        self.convt2 = nn.ConvTranspose2d(16, 3, kernel_size = 4, stride = 4)
        self.d_relu3 = nn.ReLU()
        self.d_bn2 = nn.BatchNorm2d(3)


    def encode(self, x):
        x = self.e_conv1(x)
        x = self.e_bn1(x)
        x = self.e_relu1(x)
        self.pre_flatten = x.shape
        x = self.e_flatten(x)

        mu = self.mu_fc(x)
        logvar = self.logvar_fc(x)

        return mu, logvar


    def reparametrize(self, mu, logvar):
        std = torch.exp(.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std


    def decode(self, z):
        z = self.d_lin(z)
        z = self.d_reshape(z)
        z = self.d_relu1(z)
        z = self.convt1(z)
        z = self.d_relu2(z)
        z = self.d_bn1(z)
        z = self.convt2(z)
        z = self.d_relu3(z)
        z = self.d_bn2(z)
        z = z[:, :, 4:-4, 4:-4].contiguous()

        return z


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        out = self.decode(z)

        return out, mu, logvar
