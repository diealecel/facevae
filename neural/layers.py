from torch import nn

class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        batch_size = x.shape[0]
        final_shape = tuple([batch_size] + list(self.shape))
        return x.view(final_shape)
