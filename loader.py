import os
from typing import Any, Dict, List, Tuple, Union
import argparse
from importlib import import_module
import numpy as np
#
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
#
from networks.dsp import DSP
from utils.dataset import DspArgoDataset
from utils.loss_fn import LossFn
from utils.optimizer import Optimizer
from utils.evaluator import TrajPredictionEvaluator


class Loader:
    '''
        Get and return dataset, network, loss_fn, optimizer, evaluator
    '''

    def __init__(self, args, device, is_ddp=False, world_size=1, local_rank=0):
        self.args = args
        self.device = device
        self.is_ddp = is_ddp
        self.adv_cfg = import_module('{}'.format(self.args.adv_cfg_path)).AdvCfg(is_ddp=is_ddp)
        self.world_size = world_size
        self.local_rank = local_rank
        self.resume = False

    def set_resmue(self, model_path):
        self.resume = True
        if not model_path.endswith(".tar"):
            assert False, "Model path error - '{}'".format(model_path)
        self.ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)

    def load(self):
        # dataset
        dataset = self.get_dataset()
        # network
        model = self.get_model()
        # loss_fn
        loss_fn = self.get_loss_fn()
        # optimizer
        optimizer = self.get_optimizer(model)
        # evaluator
        evaluator = self.get_evaluator()

        return dataset, model, loss_fn, optimizer, evaluator

    def get_dataset(self):
        train_dir = self.args.features_dir + 'train/'
        val_dir = self.args.features_dir + 'val/'
        test_dir = self.args.features_dir + 'test/'

        if self.args.mode == 'train' or self.args.mode == 'val':
            train_set = DspArgoDataset(train_dir,
                                       mode='train',
                                       verbose=(not self.local_rank),
                                       aug=self.args.data_aug)
            val_set = DspArgoDataset(val_dir,
                                     mode='val',
                                     verbose=(not self.local_rank))
            return train_set, val_set
        elif self.args.mode == 'test':
            test_set = DspArgoDataset(test_dir,
                                      mode='test',
                                      verbose=(not self.local_rank))
            return test_set
        else:
            assert False, "Unknown mode"

    def get_model(self):
        if self.args.model == "dsp":
            net_cfg = self.adv_cfg.get_net_cfg()
            model = DSP(net_cfg, self.device)
        else:
            assert False, "Wrong model type"

        if self.resume:
            model.load_state_dict(self.ckpt["state_dict"])

        if self.is_ddp:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(self.device)  # SyncBN
            model = model.to(self.device)
            model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)
        else:
            model = model.to(self.device)

        return model

    def get_loss_fn(self):
        if self.args.loss == "dsp":
            loss_cfg = self.adv_cfg.get_loss_cfg()
            loss = LossFn(loss_cfg, self.device)
        else:
            assert False, "Wrong loss type"

        return loss

    def get_optimizer(self, model):
        opt_cfg = self.adv_cfg.get_opt_cfg()

        if opt_cfg['lr_scale_func'] == 'linear':
            opt_cfg['lr_scale'] = self.world_size
        elif opt_cfg['lr_scale_func'] == 'sqrt':
            opt_cfg['lr_scale'] = np.round(np.sqrt(self.world_size))
        else:
            opt_cfg['lr_scale'] = 1.0

        optimizer = Optimizer(model.parameters(), opt_cfg)

        if self.resume:
            optimizer.load_state_dict(self.ckpt["opt_state"])

        return optimizer

    def get_evaluator(self):
        return TrajPredictionEvaluator()
