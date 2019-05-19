import torch
from torch import nn

from neural.layers import Flatten, Reshape

class VAE1(nn.Module):
    def __init__(self):
        super(VAE1, self).__init__()

        # self.e_cnn1_num_filters = 32
        # self.e_cnn1_kernel_size = 8
        # self.e_cnn1_padding = 2
        #
        # self.e_cnn2_num_filters = 16
        # self.e_cnn2_kernel_size = 5
        # self.e_cnn2_padding = 1
        #
        # self.maxpool_kernel_size = 2
        #
        # # Encode.
        # self.e_cnn1 = nn.Conv2d(3, self.e_cnn1_num_filters, self.e_cnn1_kernel_size, self.e_cnn1_padding)
        # self.e_maxpool = nn.MaxPool2d(self.maxpool_kernel_size, return_indices = True)
        # self.e_cnn2 = nn.Conv2d(self.e_cnn1_num_filters, self.e_cnn2_num_filters, self.e_cnn2_kernel_size, self.e_cnn2_padding)
        #
        # # Encode mu.
        # self.e_mu_fc = nn.Linear(784, 30)
        #
        # # Encode logvar.
        # self.e_logvar_fc = nn.Linear(784, 30)
        #
        #
        # # Decode
        # self.d_lin = nn.Linear(30, 784)



        # self.encode_nn_mu = nn.Sequential(
        #     nn.Conv2d(3, self.cnn1_num_filters, self.cnn1_kernel_size, self.cnn1_padding),
        #     #nn.MaxPool2d(self.maxpool_kernel_size),
        #     nn.ReLU(),
        #     nn.Conv2d(self.cnn1_num_filters, self.cnn2_num_filters, self.cnn2_kernel_size, self.cnn2_padding),
        #     nn.ReLU(),
        #     Flatten(),
        #     nn.Linear(25600, 30)
        # )
        #
        # self.encode_nn_logvar = nn.Sequential(
        #     nn.Conv2d(3, self.cnn1_num_filters, self.cnn1_kernel_size, self.cnn1_padding),
        #     #nn.MaxPool2d(self.maxpool_kernel_size),
        #     nn.ReLU(),
        #     nn.Conv2d(self.cnn1_num_filters, self.cnn2_num_filters, self.cnn2_kernel_size, self.cnn2_padding),
        #     nn.ReLU(),
        #     Flatten(),
        #     nn.Linear(25600, 30)
        # )
        #
        # self.decode_nn = nn.Sequential(
        #     nn.Linear(30, 25600),
        #     Reshape((16, 40, 40)),
        #     nn.ReLU(),
        #     nn.Conv2d(self.cnn2_num_filters, self.cnn1_num_filters, self.cnn2_kernel_size, self.cnn2_padding),
        #     #nn.MaxUnpool2d(self.maxpool_kernel_size),
        #     nn.ReLU(),
        #     nn.Conv2d(self.cnn1_num_filters, 3, self.cnn1_kernel_size, self.cnn1_padding),
        #     Flatten(),
        #     nn.Linear(507, 30000),
        #     nn.Sigmoid()
        # )

        #self.conv = nn.Conv2d(3, 32, 16, 10)

        self.e_conv1 = nn.Conv2d(3, 32, 16, 10)
        self.e_bn1 = nn.BatchNorm2d(32)
        self.mu_fc = nn.Linear(2592, 1000)
        self.logvar_fc = nn.Linear(2592, 1000)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


        self.d_lin = nn.Linear(1000, 2592)
        self.up = nn.UpsamplingNearest2d(scale_factor = 4)
        self.up2 = nn.UpsamplingNearest2d(scale_factor = 3)
        self.d_conv = nn.Conv2d(32, 3, 9, 1)
        self.d_bn1 = nn.BatchNorm2d(3)




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
        x = self.e_bn1(x)
        x = self.relu(x)
        self.pre_flatten = x.shape
        x = Flatten()(x)

        mu = self.mu_fc(x)
        logvar = self.logvar_fc(x)

        return mu, logvar


    def reparametrize(self, mu, logvar):
        std = torch.exp(.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std


    def decode(self, z):
        # z = nn.Linear(500, 1000)(z)
        # z = nn.ReLU()(z)
        # z = nn.Linear(1000, 2000)(z)
        # z = nn.ReLU()(z)
        # z = nn.Linear(2000, 5776)(z)
        # z = Reshape(self.pre_flat_shape)(z)
        # z = nn.ReLU()(z)
        # z = nn.UpsamplingNearest2d(scale_factor = 2)(z)
        # z = nn.ReplicationPad2d(2)(z)
        # z = nn.Conv2d(16, 32, 2, 2, 2)(z)
        # z = nn.BatchNorm2d(32)(z)
        # z = nn.MaxUnpool2d(self.maxpool_kernel_size)(z, self.max_indices, output_size = self.whataburger)
        # z = nn.UpsamplingNearest2d(scale_factor = 2)(z)
        # z = nn.ReplicationPad2d(3)(z)
        # z = nn.Conv2d(32, 3, 5, 1, 2)(z)
        # z = nn.BatchNorm2d(3)(z)
        # z = Flatten()(z)
        # z = nn.Sigmoid()(z)

        z = self.d_lin(z)
        z = Reshape(self.pre_flatten[1:])(z)
        z = self.relu(z)
        z = self.up(z)
        z = self.relu(z)
        z = self.up2(z)
        z = self.d_conv(z)
        z = self.d_bn1(z)
        z = Flatten()(z)
        z = self.sigmoid(z)

        #print(z.shape)

        return z


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        out = self.decode(z)

        return out, mu, logvar
