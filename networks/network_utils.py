import math
import os
import pickle as pkl
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
#
import torch
import torch.nn as nn


def conv1d(in_channels,
           out_channels,
           kernel_size,
           stride=1,
           padding=0,
           dilation=1,
           apply_bn=True,
           activation=None):
    net = [
        nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding,
                  dilation)
    ]
    if apply_bn:
        net.append(nn.BatchNorm1d(out_channels))

    if activation is not None:
        if activation == 'relu':
            net.append(nn.ReLU())
        else:
            assert False, "invalid activation type"
    return nn.Sequential(*net)


def linear(in_size, out_size, apply_bn=True, activation=None):
    net = [nn.Linear(in_size, out_size)]
    if apply_bn:
        net.append(nn.BatchNorm1d(out_size))

    if activation is not None:
        if activation == 'relu':
            net.append(nn.ReLU())
        elif activation == 'leaky_relu':
            net.append(nn.LeakyReLU(0.1))
        else:
            assert False, "invalid activation type"
    return nn.Sequential(*net)


# Custom activation for output layer (Graves, 2015)
# Generating Sequences With Recurrent Neural Networks, Alex Graves, Eq. 20-22
def prob_traj_output(x):
    # e.g., [batch, seq, 1] or # [batch, n_dec, seq, 1]
    muX = x[..., 0:1]  # [..., 1]
    muY = x[..., 1:2]  # [..., 1]
    sigX = x[..., 2:3]  # [..., 1]
    sigY = x[..., 3:4]  # [..., 1]
    rho = x[..., 4:5]  # [..., 1]
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)

    out = torch.cat([muX, muY, sigX, sigY, rho], dim=-1)  # [..., 5]
    return out
