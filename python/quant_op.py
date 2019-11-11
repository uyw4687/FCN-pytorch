from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch 
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class FakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bound, n_lv):
        diff = bound / (n_lv - 1)
        return input.div(diff).round_().clamp_(0, n_lv-1).mul_(diff)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class Q_ReLU(nn.ReLU):
    def __init__(self, *args, **kargs):
        super(Q_ReLU, self).__init__(*args, **kargs)
        self.n_lv = 0
        self.bound = 0
    def forward(self, x):
        if self.bound == 0:
            return F.relu(x)
        else:
            x = F.hardtanh(x, 0, self.bound)

            if self.n_lv > 0:
                x = FakeQuant.apply(x, self.bound, self.n_lv)
            return x

    
class Q_Conv2d(nn.Conv2d):
    def __init__(self, *args, **kargs):
        super(Q_Conv2d, self).__init__(*args, **kargs)
        self.n_lv = 0
        self.ratio = 0

    def forward(self, x):
        if self.n_lv == 0:
            return F.conv2d(x, self.weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups)
        else:
            sign = self.weight.sign()
            weight = self.weight.abs()
            bound = weight.max() * self.ratio
            weight = sign * FakeQuant.apply(weight, bound, self.n_lv // 2)

            return F.conv2d(x, weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups)

    
class Q_ConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, *args, **kargs):
        super(Q_ConvTranspose2d, self).__init__(*args, **kargs)
        self.n_lv = 0
        self.ratio = 0

    def forward(self, x):
        if self.n_lv == 0:
            return F.conv_transpose2d(x, self.weight, self.bias,
                self.stride, self.padding, self.output_padding, self.groups, self.dilation)
        else:
            sign = self.weight.sign()
            weight = self.weight.abs()
            bound = weight.max() * self.ratio
            weight = sign * FakeQuant.apply(weight, bound, self.n_lv // 2)

            return F.conv_transpose2d(x, weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups)