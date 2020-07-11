from .layer import *
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from base import BaseModel

class Vd_Mlp(nn.Module):
    """1 hidden layer Variational Dropout Network"""
    def __init__(self, input_dim, num_classes, n_hid, alpha_shape=(1, 1), bias=True, activation='ReLU'):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.n_hid = n_hid
        self.alpha_shape = alpha_shape
        self.bias = bias

        fc = []
        fc.append(VdLinear(self.input_dim, self.n_hid[0], alpha_shape, bias))
        for i in range(1, len(self.n_hid)):
            fc.append(VdLinear(self.n_hid[i-1], self.n_hid[i], alpha_shape, bias))
        fc.append(VdLinear(self.n_hid[-1], self.num_classes, alpha_shape, bias))

        self.fc = nn.ModuleList(fc)

        # Non linearity
        if activation == 'ReLU':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'Tanh':
            self.act = nn.Tanh()
        elif activation == 'Sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x, sample=False):
        tkl = 0.0

        x = x.view(-1, self.input_dim)  # view(batch_size, input_dim)

        for i in range(len(self.fc) - 1):
            # -----------------
            x, kl = self.fc[i](x, sample)
            tkl = tkl + kl
            # -----------------
            x = self.act(x)
            # -----------------
        y, kl = self.fc[-1](x, sample)
        tkl = tkl + kl

        return y, tkl

    def sample_predict(self, x, Nsamples):
        """Used for estimating the data's likelihood by approximately marginalising the weights with MC"""
        # Just copies type from x, initializes new vector
        predictions = x.data.new(Nsamples, x.shape[0], self.output_dim)
        tkl_vec = np.zeros(Nsamples)

        for i in range(Nsamples):
            y, tkl = self.forward(x, sample=True)
            predictions[i] = y
            tkl_vec[i] = tkl

        return predictions, tkl_vec

class Vd_CNN(nn.Module):
    """ Variational Dropout CNN Network"""
    def __init__(self, input_dim, num_classes, filters, kernels, alpha_shape=(1, 1), bias=True, activation='ReLU'):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.filters = filters
        self.kernels = kernels
        self.alpha_shape = alpha_shape
        self.bias = bias

        num_features = 256*125
        conv = []
        conv.append(VdConv1D(self.input_dim, self.filters[0], self.kernels[0], alpha_shape, bias))
        for i in range(1, len(self.filters)):
            conv.append(VdConv1D(self.filters[i - 1], self.filters[i], self.kernels[i], alpha_shape, bias))
        self.conv = nn.ModuleList(conv)
        self.fc = VdLinear(num_features, self.num_classes, alpha_shape, bias)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)

        # Non linearity
        if activation == 'ReLU':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'Tanh':
            self.act = nn.Tanh()
        elif activation == 'Sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x, sample=False):
        tkl = 0.0

        for i in range(len(self.conv)):
            # -----------------
            x, kl = self.conv[i](x, sample)
            tkl = tkl + kl
            # -----------------
            x = self.act(x)
            x = self.pool(x)
            # -----------------
        x = x.reshape(x.size(0), -1)

        y, kl = self.fc(x, sample)
        tkl = tkl + kl

        return y, tkl

