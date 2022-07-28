import os
import sys


class AdvCfg():
    def __init__(self, is_ddp=False):
        self.is_ddp = is_ddp
        print('[AdvCfg] is_ddp: ', self.is_ddp)

    def get_net_cfg(self):
        net_cfg = {}

        net_cfg["n_da"] = 32
        net_cfg["n_scales_da"] = 4

        net_cfg["n_actor"] = 128
        net_cfg["n_ls"] = 128
        net_cfg["n_scales_ls"] = 4
        net_cfg["a2ls_dist"] = 7.0

        net_cfg["map2actor_dist"] = 6.0
        net_cfg["actor2actor_dist"] = 100.0

        net_cfg["n_mode"] = 6

        net_cfg["p_dropout"] = 0.1

        return net_cfg

    def get_loss_cfg(self):
        loss_cfg = {}
        loss_cfg['w_reg'] = 0.2
        loss_cfg['w_goal'] = 0.8

        loss_cfg['w_goal_cls'] = 0.8
        loss_cfg['w_goal_reg'] = 0.2

        loss_cfg['fl_alpha'] = 2.0
        loss_cfg['fl_beta'] = 4.0
        loss_cfg['fl_sigma'] = 0.5
        loss_cfg['pos_thres'] = 0.85  # thres: 1m -> 0.88

        return loss_cfg

    def get_opt_cfg(self):
        opt_cfg = {}
        opt_cfg['opt'] = 'adam'
        opt_cfg['lr_scale_func'] = 'sqrt'  # sqrt/linear
        opt_cfg['lr_func'] = 'poly'
        opt_cfg['lr_func_cfg'] = {}
        opt_cfg['lr_func_cfg']['default_lr'] = 1e-3

        if self.is_ddp:
            opt_cfg['lr_func_cfg']['epochs'] = [0, 5, 25, 30]
            opt_cfg['lr_func_cfg']['lrs'] = [1e-5, 1e-4, 1e-4, 1e-5]
        else:
            opt_cfg['lr_func_cfg']['epochs'] = [0, 5, 25, 30]
            opt_cfg['lr_func_cfg']['lrs'] = [1e-5, 1e-4, 1e-4, 1e-5]

        return opt_cfg
