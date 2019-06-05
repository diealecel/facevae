import torch
from torch import nn
from torch.distributions.normal import Normal

from neural.layers import Flatten, Reshape

class VAE6_1(nn.Module):
    def __init__(self, save_idx_dist = False):
        super(VAE6_1, self).__init__()

        self._save_idx_dist = save_idx_dist
        self._use_idx_dist = False
        self._maxpool1_idx_mean, self._maxpool1_idx_std = None, None
        self._maxpool2_idx_mean, self._maxpool2_idx_std = None, None
        self._maxpool3_idx_mean, self._maxpool3_idx_std = None, None


        # ENCODE
        self.e_conv1 = nn.Conv2d(3, 64, kernel_size = 3)
        self.e_prelu1 = nn.PReLU()
        self.e_bn1 = nn.BatchNorm2d(64)
        self.e_maxpool1 = nn.MaxPool2d(kernel_size = 2, return_indices = True)
        self.e_conv2 = nn.Conv2d(64, 64, kernel_size = 3)
        self.e_prelu2 = nn.PReLU()
        self.e_bn2 = nn.BatchNorm2d(64)
        self.e_maxpool2 = nn.MaxPool2d(kernel_size = 2, return_indices = True)
        self.e_conv3 = nn.Conv2d(64, 32, kernel_size = 2)
        self.e_prelu3 = nn.PReLU()
        self.e_bn3 = nn.BatchNorm2d(32)
        self.e_maxpool3 = nn.MaxPool2d(kernel_size = 2, return_indices = True)
        self.e_conv4 = nn.Conv2d(32, 32, kernel_size = 2)
        self.e_prelu4 = nn.PReLU()
        self.e_bn4 = nn.BatchNorm2d(32)
        self.e_conv5 = nn.Conv2d(32, 16, kernel_size = 2)
        self.e_prelu5 = nn.PReLU()
        self.e_bn5 = nn.BatchNorm2d(16)
        self.e_conv6 = nn.Conv2d(16, 16, kernel_size = 2)
        self.e_prelu6 = nn.PReLU()
        self.e_bn6 = nn.BatchNorm2d(16)
        self.e_conv7 = nn.Conv2d(16, 8, kernel_size = 2)
        self.e_prelu7 = nn.PReLU()
        self.e_bn7 = nn.BatchNorm2d(8)
        self.e_flatten = Flatten()
        self.e_lin1 = nn.Linear(392, 200)
        self.e_prelu8 = nn.PReLU()

        self.mu_fc = nn.Linear(200, 1)
        self.logvar_fc = nn.Linear(200, 1)


        # DECODE
        self.d_lin1 = nn.Linear(1, 200)
        self.d_prelu1 = nn.PReLU()
        self.d_lin2 = nn.Linear(200, 392)
        self.d_reshape = Reshape((8, 7, 7))
        self.d_prelu2 = nn.PReLU()
        self.d_convt1 = nn.ConvTranspose2d(8, 16, kernel_size = 2)
        self.d_prelu3 = nn.PReLU()
        self.d_bn1 = nn.BatchNorm2d(16)
        self.d_convt2 = nn.ConvTranspose2d(16, 16, kernel_size = 2)
        self.d_prelu4 = nn.PReLU()
        self.d_bn2 = nn.BatchNorm2d(16)
        self.d_convt3 = nn.ConvTranspose2d(16, 32, kernel_size = 2)
        self.d_prelu5 = nn.PReLU()
        self.d_bn3 = nn.BatchNorm2d(32)
        self.d_convt4 = nn.ConvTranspose2d(32, 32, kernel_size = 2)
        self.d_prelu6 = nn.PReLU()
        self.d_bn4 = nn.BatchNorm2d(32)
        self.d_maxunpool1 = nn.MaxUnpool2d(kernel_size = 2)
        self.d_convt5 = nn.ConvTranspose2d(32, 64, kernel_size = 2)
        self.d_prelu7 = nn.PReLU()
        self.d_bn5 = nn.BatchNorm2d(64)
        self.d_maxunpool2 = nn.MaxUnpool2d(kernel_size = 2)
        self.d_convt6 = nn.ConvTranspose2d(64, 64, kernel_size = 3)
        self.d_prelu8 = nn.PReLU()
        self.d_bn6 = nn.BatchNorm2d(64)
        self.d_maxunpool3 = nn.MaxUnpool2d(kernel_size = 2)
        self.d_convt7 = nn.ConvTranspose2d(64, 3, kernel_size = 3)
        self.d_bn7 = nn.BatchNorm2d(3)


    def record_idx(self, idxs, layer):
        if self._save_idx_dist:
            with torch.no_grad():
                idxs_float = idxs.type(torch.float)
                idxs_mean = idxs_float.mean(dim = 0)
                idxs_std = idxs_float.std(dim = 0)

                if layer == 1:
                    if self._maxpool1_idx_mean is None:
                        self._maxpool1_idx_mean = idxs_mean
                        self._maxpool1_idx_std = idxs_std
                    else:
                        self._maxpool1_idx_mean = (self._maxpool1_idx_mean + idxs_mean) / 2
                        self._maxpool1_idx_std = (self._maxpool1_idx_std + idxs_std) / 2

                if layer == 2:
                    if self._maxpool2_idx_mean is None:
                        self._maxpool2_idx_mean = idxs_mean
                        self._maxpool2_idx_std = idxs_std
                    else:
                        self._maxpool2_idx_mean = (self._maxpool2_idx_mean + idxs_mean) / 2
                        self._maxpool2_idx_std = (self._maxpool2_idx_std + idxs_std) / 2

                if layer == 3:
                    if self._maxpool3_idx_mean is None:
                        self._maxpool3_idx_mean = idxs_mean
                        self._maxpool3_idx_std = idxs_std
                    else:
                        self._maxpool3_idx_mean = (self._maxpool3_idx_mean + idxs_mean) / 2
                        self._maxpool3_idx_std = (self._maxpool3_idx_std + idxs_std) / 2


    def activate_idx_sampling(self):
        self._maxpool1_idx_dist = Normal(loc = self._maxpool1_idx_mean, scale = self._maxpool1_idx_std)
        self._maxpool2_idx_dist = Normal(loc = self._maxpool2_idx_mean, scale = self._maxpool2_idx_std)
        self._maxpool3_idx_dist = Normal(loc = self._maxpool3_idx_mean, scale = self._maxpool3_idx_std)
        self._use_idx_dist = True


    def encode(self, x):
        x = self.e_conv1(x)
        x = self.e_prelu1(x)
        x = self.e_bn1(x)
        self._premaxpool1_shape = x.shape
        x, self._maxpool1_idx = self.e_maxpool1(x)
        self.record_idx(self._maxpool1_idx, layer = 1)
        x = self.e_conv2(x)
        x = self.e_prelu2(x)
        x = self.e_bn2(x)
        self._premaxpool2_shape = x.shape
        x, self._maxpool2_idx = self.e_maxpool2(x)
        self.record_idx(self._maxpool2_idx, layer = 2)
        x = self.e_conv3(x)
        x = self.e_prelu3(x)
        x = self.e_bn3(x)
        self._premaxpool3_shape = x.shape
        x, self._maxpool3_idx = self.e_maxpool3(x)
        self.record_idx(self._maxpool3_idx, layer = 3)
        x = self.e_conv4(x)
        x = self.e_prelu4(x)
        x = self.e_bn4(x)
        x = self.e_conv5(x)
        x = self.e_prelu5(x)
        x = self.e_bn5(x)
        x = self.e_conv6(x)
        x = self.e_prelu6(x)
        x = self.e_bn6(x)
        x = self.e_conv7(x)
        x = self.e_prelu7(x)
        x = self.e_bn7(x)
        x = self.e_flatten(x)
        x = self.e_lin1(x)
        x = self.e_prelu8(x)

        mu = self.mu_fc(x)
        logvar = self.logvar_fc(x)

        return mu, logvar


    def reparametrize(self, mu, logvar):
        std = torch.exp(.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std


    def decode(self, z):
        z = self.d_lin1(z)
        z = self.d_prelu1(z)
        z = self.d_lin2(z)
        z = self.d_reshape(z)
        z = self.d_prelu2(z)
        z = self.d_convt1(z)
        z = self.d_prelu3(z)
        z = self.d_bn1(z)
        z = self.d_convt2(z)
        z = self.d_prelu4(z)
        z = self.d_bn2(z)
        z = self.d_convt3(z)
        z = self.d_prelu5(z)
        z = self.d_bn3(z)
        z = self.d_convt4(z)
        z = self.d_prelu6(z)
        z = self.d_bn4(z)
        maxpool3_idx = self._maxpool3_idx_dist.sample().type(torch.long).unsqueeze_(0) if self._use_idx_dist else self._maxpool3_idx
        z = self.d_maxunpool1(z, maxpool3_idx, output_size = self._premaxpool3_shape)
        z = self.d_convt5(z)
        z = self.d_prelu7(z)
        z = self.d_bn5(z)
        maxpool2_idx = self._maxpool2_idx_dist.sample().type(torch.long).unsqueeze_(0) if self._use_idx_dist else self._maxpool2_idx
        z = self.d_maxunpool2(z, maxpool2_idx, output_size = self._premaxpool2_shape)
        z = self.d_convt6(z)
        z = self.d_prelu8(z)
        z = self.d_bn6(z)
        maxpool1_idx = self._maxpool1_idx_dist.sample().type(torch.long).unsqueeze_(0) if self._use_idx_dist else self._maxpool1_idx
        z = self.d_maxunpool3(z, maxpool1_idx, output_size = self._premaxpool1_shape)
        z = self.d_convt7(z)
        z = self.d_bn7(z)

        return z


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        out = self.decode(z)

        return out, mu, logvar
