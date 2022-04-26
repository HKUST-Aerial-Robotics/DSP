import os
from typing import Any, Dict, List, Optional, Tuple, Union
#
import numpy as np
#
import torch


def from_numpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data)
    return data


def gpu(data, device):
    """
    Transfer tensor in `data` to gpu recursively
    `data` can be dict, list or tuple
    """
    if isinstance(data, list) or isinstance(data, tuple):
        data = [gpu(x, device=device) for x in data]
    elif isinstance(data, dict):
        data = {key: gpu(_data, device=device) for key, _data in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.contiguous().to(device, non_blocking=True)
    return data


def to_long(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_long(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_long(x) for x in data]
    if torch.is_tensor(data) and data.dtype == torch.int16:
        data = data.long()
    return data


def check_loss_abnormal(loss_hist, scale=2.0):
    # TODO return info
    return False


def str2bool(v):
    if v.lower() in ('yes', 'True', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_tensor_memory(data):
    return data.element_size() * data.nelement() / 1024 / 1024


def save_ckpt(net, opt, epoch, save_dir, filename):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    torch.save(
        {"epoch": epoch, "state_dict": state_dict, "opt_state": opt.opt.state_dict()},
        os.path.join(save_dir, filename),
    )


def load_pretrain(net, pretrain_dict):
    state_dict = net.state_dict()
    for key in pretrain_dict.keys():
        if key in state_dict and (pretrain_dict[key].size() == state_dict[key].size()):
            value = pretrain_dict[key]
            if not isinstance(value, torch.Tensor):
                value = value.data
            state_dict[key] = value
    net.load_state_dict(state_dict)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.detach().cpu().item()
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeterForDict(object):
    def __init__(self):
        self.reset()       # __init__():reset parameters

    def reset(self):
        self.metrics = {}

    def update(self, elem, n=1):
        for key, val in elem.items():
            if not key in self.metrics:
                self.metrics[key] = AverageMeter()

            self.metrics[key].update(val, n)

    def get_info(self):
        info = ''
        for key, elem in self.metrics.items():
            info += "{}: {:.3f} ".format(key, elem.avg)
        return info

    def print(self):
        info = self.get_info()
        print('-- ' + info)


class ScalarMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.vals = []

    def push(self, val):
        self.vals.append(val)

    def mean(self):
        return np.mean(self.vals)

    def max(self):
        return np.max(self.vals)

    def min(self):
        return np.min(self.vals)


class ScalarMeterForDict(object):
    def __init__(self):
        self.reset()       # __init__():reset parameters

    def reset(self):
        self.metrics = {}

    def push(self, elem):
        for key, val in elem.items():
            if not key in self.metrics:
                self.metrics[key] = ScalarMeter()

            self.metrics[key].push(val)

    def get_info(self):
        info = ''
        for key, elem in self.metrics.items():
            info += "{}: [{:.3f} {:.3f} {:.3f}] ".format(key, elem.min(), elem.mean(), elem.max())
        return info

    def print(self):
        info = self.get_info()
        print('-- ' + info)
