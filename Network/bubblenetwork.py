# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class DilatedCNN(torch.nn.Module):
    def __init__(self, hidden_size=64, depth=12):
        super(DilatedCNN, self).__init__()
        model = []
        model += [nn.ReflectionPad1d(2**depth - 1)]
        model += [nn.Conv1d(1, hidden_size, kernel_size=3, dilation=1)]
        model += [nn.BatchNorm1d(hidden_size)]
        model += [nn.ReLU()]      
        for l in range(1, depth):
            model += [nn.Conv1d(hidden_size, hidden_size, kernel_size=3, dilation=2**l)]
            model += [nn.BatchNorm1d(hidden_size)]
            model += [nn.ReLU()]    
        model += [nn.Conv1d(hidden_size, 1, kernel_size=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        z = self.model(x)

        return z
    
class RandomModel(torch.nn.Module):
    # This model outputs random numbers between 0 and 1, with the same shape as
    # the input batch:
    def __init__(self):
        super(RandomModel, self).__init__()
        self.model = []

    def forward(self, x):
        return torch.rand(x.shape).cuda()