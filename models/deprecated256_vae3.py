import torch
from torch import nn

from neural.layers import Flatten, Reshape

class VAE4(nn.Module):
    def __init__(self):
        super(VAE4, self).__init__()

        # ENCODE
        self.e_conv1 = nn.Conv2d(3, 32, kernel_size = 3)
        # ReLU
        self.e_bn1 = nn.BatchNorm2d(32)
        self.e_maxpool1 = nn.MaxPool2d(kernel_size = 4, return_indices = True)
        self.e_conv2 = nn.Conv2d(32, 16, kernel_size = 5, stride = 4)
        # ReLU
        # Flatten
        self.e_lin1 = nn.Linear(3600, 1000)
        # ReLU

        self.mu_fc = nn.Linear(1000, 500)
        self.logvar_fc = nn.Linear(1000, 500)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.flatten = Flatten()

        # DECODE
        self.d_lin1 = nn.Linear(500, 1000)
        # ReLU
        self.d_lin2 = nn.Linear(1000, 3600)
        # Reshape
        # ReLU
        self.d_convt1 = nn.ConvTranspose2d(16, 32, kernel_size = 5, stride = 4, output_padding = 2)
        # ReLU
        self.d_bn1 = nn.BatchNorm2d(32)
        self.d_maxunpool1 = nn.MaxUnpool2d(kernel_size = 4)
        self.d_convt2 = nn.ConvTranspose2d(32, 3, kernel_size = 3)
        # ReLU
        self.d_bn2 = nn.BatchNorm2d(3)



    def encode(self, x):
        # x = self.e_cnn1(x)
        # x = nn.BatchNorm2d(32)(x)
        # self.whataburger = x.shape
        # x, self.max_indices = self.e_maxpool(x)
        # x = self.e_cnn2(x)
        # x = nn.BatchNorm2d(16)(x)
        # x = nn.ReLU()(x)
        # self.pre_flat_shape = x.shape[1:]
        # x = Flatten()(x)
        # x = nn.Linear(5776, 2000)(x)
        # x = nn.ReLU()(x)
        # x = nn.Linear(2000, 1000)(x)
        # x = nn.ReLU()(x)
        # mu = nn.Linear(1000, 500)(x)
        # logvar = nn.Linear(1000, 500)(x)


        #x = self.conv(x)
        x = self.e_conv1(x)
        x = self.relu(x)
        x = self.e_bn1(x)
        self._premaxpool_shape = x.shape
        x, self.idx = self.e_maxpool1(x)
        #print(x.shape)
        x = self.e_conv2(x)
        x = self.relu(x)
        self._preflatten_shape = x.shape
        x = self.flatten(x)
        x = self.e_lin1(x)
        x = self.relu(x)

        mu = self.mu_fc(x)
        logvar = self.logvar_fc(x)

        return mu, logvar


    def reparametrize(self, mu, logvar):
        std = torch.exp(.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std


    def decode(self, z):
        z = self.d_lin1(z)
        z = self.relu(z)
        z = self.d_lin2(z)
        z = Reshape(self._preflatten_shape[1:])(z)
        z = self.relu(z)
        z = self.d_convt1(z)
        z = self.relu(z)
        #print(z.shape)
        z = self.d_maxunpool1(z, self.idx, output_size = self._premaxpool_shape)
        z = self.d_convt2(z)
        z = self.relu(z)
        z = self.d_bn2(z)
        z = Flatten()(z)
        z = self.sigmoid(z)

        return z


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        out = self.decode(z)

        return out, mu, logvar
