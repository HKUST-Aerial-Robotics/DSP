import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import gpu, to_long
from utils.loss_fn_utils import (reg_nll_loss,
                                 reg_huber_loss,
                                 variant_focal_bce_loss)


class LossFn(nn.Module):
    def __init__(self, config, device):
        super(LossFn, self).__init__()
        self.config = config
        self.device = device

    def forward(self, out, data):
        traj_pred = out['traj_pred']  # batch x n_mod x seq x 5
        goda_pred = out['goda_cls']
        goal_pred = out['goal_pred']  # (batch, n_mode, 2)

        traj_fut = torch.stack([traj[0, :, 0:2] for traj in data['TRAJS_FUT']]).to(self.device)  # batch x fut x 2
        goal_gt = gpu(torch.stack(data['GOAL_GT']), device=self.device)  # (batch, 2)

        # ~ Goal
        # cls
        goda_label = torch.cat(gpu(data['GODA_LABEL'], device=self.device))
        loss_goal_cls = variant_focal_bce_loss(goda_pred, goda_label,
                                               alpha=self.config['fl_alpha'],
                                               beta=self.config['fl_beta'],
                                               thres=self.config['pos_thres'],
                                               sigma=self.config['fl_sigma'])
        # thres: 1m -> 0.88

        # reg
        loss_goal_reg = self.calc_wta_goal_reg_loss(goal_pred, goal_gt)

        loss_goal = loss_goal_cls * self.config['w_goal_cls'] + loss_goal_reg * self.config['w_goal_reg']

        # ~ Trajectory
        # traj regression
        loss_reg = self.calc_wta_reg_loss(traj_pred, traj_fut)

        loss_out = {}
        loss_out['loss_reg'] = loss_reg * self.config['w_reg']
        loss_out['loss_goal'] = loss_goal * self.config['w_goal']
        loss_out['loss'] = loss_out['loss_reg'] + loss_out['loss_goal']
        return loss_out

    def calc_wta_reg_loss(self, traj_pred, traj_fut):
        # traj_pred:    batch x n_mod x seq x 5
        _traj_fut = torch.stack([traj_fut for _ in range(traj_pred.shape[1])], dim=1)

        loss_reg = reg_nll_loss(traj_pred, _traj_fut)
        _, min_idcs = loss_reg.min(1)
        row_idcs = torch.arange(len(min_idcs)).long().to(self.device)
        # traj reg
        loss_reg = loss_reg[row_idcs, min_idcs]  # batch x fut x 2
        return torch.mean(loss_reg)

    def calc_wta_goal_reg_loss(self, goal_pred, goal_gt):
        # goal_pred:    batch x n_mode x 2
        # goal_gt:      batch x 2
        _goal_gt = torch.stack([goal_gt for _ in range(goal_pred.shape[1])], dim=1)

        loss = reg_huber_loss(goal_pred, _goal_gt)  # batch x n_mode
        _, min_idcs = loss.min(1)
        row_idcs = torch.arange(len(min_idcs)).long().to(self.device)
        loss = loss[row_idcs, min_idcs]  # batch
        return torch.mean(loss)

    def print(self):
        print('\nloss_fn config:', self.config)
        print('loss_fn device:', self.device)
