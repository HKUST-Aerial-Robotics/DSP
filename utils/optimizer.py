import os
from typing import Any, Dict, List, Tuple
#
import numpy as np
import torch


class ConstLR:
    def __init__(self, lr):
        self.lr = lr

    def __call__(self, epoch):
        return self.lr


class StepLR:
    def __init__(self, lrs, lr_epochs):
        assert len(lrs) - len(lr_epochs) == 1
        self.lrs = lrs
        self.lr_epochs = lr_epochs

    def __call__(self, epoch):
        idx = 0
        for lr_epoch in self.lr_epochs:
            if epoch < lr_epoch:
                break
            idx += 1
        return self.lrs[idx]


class PolylineLR():
    def __init__(self, epochs, lrs):
        assert len(epochs) == len(lrs), 'length must be same'
        assert np.all(np.array(lrs) > 0), 'lrs must be positive'
        assert np.all(np.array(epochs) >= 0), 'epochs must be positive'
        assert epochs[0] == 0, 'epochs must start from 0'
        assert epochs == sorted(epochs), 'epochs must be in ascending order'

        self.epochs = epochs  # x
        self.lrs = lrs  # y
        self.n_intervals = len(self.epochs) - 1

    def __call__(self, epoch):
        assert epoch >= 0
        for i in range(self.n_intervals):
            e_lb = self.epochs[i]
            e_ub = self.epochs[i + 1]

            if epoch < e_lb or epoch >= e_ub:
                continue  # not in this interval

            v_lb = self.lrs[i]
            v_ub = self.lrs[i + 1]

            v_e = (epoch - e_lb) / (e_ub - e_lb) * (v_ub - v_lb) + v_lb
            return v_e

        return self.lrs[-1]


class Optimizer(object):
    def __init__(self, net_params, config):

        self.lr_func = None
        # scale factor for DDP
        self.lr_scale = config['lr_scale'] if 'lr_scale' in config else 1.0
        self.config = config

        # lr_func
        if config["lr_func"] == 'poly':
            epochs = config["lr_func_cfg"]["epochs"]
            lrs = config["lr_func_cfg"]["lrs"]
            self.lr_func = PolylineLR(epochs, lrs)
        elif config["lr_func"] == 'step':
            epochs = config["lr_func_cfg"]["epochs"]
            lrs = config["lr_func_cfg"]["lrs"]
            self.lr_func = StepLR(lrs, epochs)
        elif config["lr_func"] == 'const':
            lr = config["lr_func_cfg"]["default_lr"]
            self.lr_func = ConstLR(lr)
        else:
            assert False, 'unknown lr func, should be poly/step/const'

        # optimizer
        opt = config["opt"]
        if opt == "sgd":
            self.opt = torch.optim.SGD(net_params)
        elif opt == "adam":
            self.opt = torch.optim.Adam(net_params)
        else:
            assert False, 'unknown opt type, should be sgd/adam'

    def zero_grad(self):
        self.opt.zero_grad()

    def step(self, epoch):
        scaled_lr = self.lr_func(epoch) * self.lr_scale
        assert scaled_lr > 0.0, 'wrong learning rate {}'.format(scaled_lr)

        for param_group in self.opt.param_groups:
            param_group["lr"] = scaled_lr
        self.opt.step()
        return scaled_lr

    def load_state_dict(self, opt_state):
        self.opt.load_state_dict(opt_state)

    def print(self):
        print('\noptimizer config:', self.config)
