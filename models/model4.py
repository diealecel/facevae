import torch
from torch import nn

from neural.layers import Flatten, Reshape

class VAE4(nn.Module):
    def __init__(self):
        super(VAE4, self).__init__()

        # ENCODE
        self.e_conv1 = nn.Conv2d(3, 64, kernel_size = 3)
        self.e_relu1 = nn.ReLU()
        self.e_bn1 = nn.BatchNorm2d(64)
        self.e_maxpool1 = nn.MaxPool2d(kernel_size = 2, return_indices = True)
        self.e_conv2 = nn.Conv2d(64, 32, kernel_size = 4)
        self.e_relu2 = nn.ReLU()
        self.e_bn2 = nn.BatchNorm2d(32)
        self.e_maxpool2 = nn.MaxPool2d(kernel_size = 3, return_indices = True)
        self.e_conv3 = nn.Conv2d(32, 16, kernel_size = 4)
        self.e_relu3 = nn.ReLU()
        self.e_bn3 = nn.BatchNorm2d(16)
        self.e_flatten = Flatten()
        self.e_lin1 = nn.Linear(2304, 1000)
        self.e_relu4 = nn.ReLU()

        self.mu_fc = nn.Linear(1000, 500)
        self.logvar_fc = nn.Linear(1000, 500)


        # DECODE
        self.d_lin1 = nn.Linear(500, 1000)
        self.d_relu1 = nn.ReLU()
        self.d_lin2 = nn.Linear(1000, 2304)
        self.d_reshape = Reshape((16, 12, 12))
        self.d_relu2 = nn.ReLU()
        self.d_convt1 = nn.ConvTranspose2d(16, 32, kernel_size = 4)
        self.d_bn1 = nn.BatchNorm2d(32)
        self.d_maxunpool1 = nn.MaxUnpool2d(kernel_size = 3)
        self.d_convt2 = nn.ConvTranspose2d(32, 64, kernel_size = 4)
        self.d_relu3 = nn.ReLU()
        self.d_bn2 = nn.BatchNorm2d(64)
        self.d_maxunpool2 = nn.MaxUnpool2d(kernel_size = 2)
        self.d_convt3 = nn.ConvTranspose2d(64, 3, kernel_size = 3)
        self.d_relu4 = nn.ReLU()
        self.d_bn3 = nn.BatchNorm2d(3)


    def encode(self, x):
        x = self.e_conv1(x)
        x = self.e_relu1(x)
        x = self.e_bn1(x)
        self._premaxpool1_shape = x.shape
        x, self._maxpool1_idx = self.e_maxpool1(x)
        x = self.e_conv2(x)
        x = self.e_relu2(x)
        x = self.e_bn2(x)
        self._premaxpool2_shape = x.shape
        x, self._maxpool2_idx = self.e_maxpool2(x)
        x = self.e_conv3(x)
        x = self.e_relu3(x)
        x = self.e_bn3(x)
        x = self.e_flatten(x)
        x = self.e_lin1(x)
        x = self.e_relu4(x)

        mu = self.mu_fc(x)
        logvar = self.logvar_fc(x)

        return mu, logvar


    def reparametrize(self, mu, logvar):
        std = torch.exp(.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std


    def decode(self, z):
        z = self.d_lin1(z)
        z = self.d_relu1(z)
        z = self.d_lin2(z)
        z = self.d_reshape(z)
        z = self.d_relu2(z)
        z = self.d_convt1(z)
        z = self.d_bn1(z)
        z = self.d_maxunpool1(z, self._maxpool2_idx, output_size = self._premaxpool2_shape)
        z = self.d_convt2(z)
        z = self.d_relu3(z)
        z = self.d_bn2(z)
        z = self.d_maxunpool2(z, self._maxpool1_idx, output_size = self._premaxpool1_shape)
        z = self.d_convt3(z)
        z = self.d_relu4(z)
        z = self.d_bn3(z)

        return z


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        out = self.decode(z)

        return out, mu, logvar
