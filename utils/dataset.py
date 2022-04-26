import os
import math
import time
import random
from typing import Any, Dict, List, Optional, Tuple, Union
#
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import spatial
#
import torch
import torch.nn as nn
from torch.utils.data import Dataset
#
from utils.utils import from_numpy


class DspArgoDataset(Dataset):
    def __init__(self,
                 dataset_dir: str,
                 mode: str,
                 obs_len: int = 20,
                 pred_len: int = 30,
                 aug: bool = False,
                 verbose: bool = False):
        self.mode = mode
        self.aug = aug
        self.verbose = verbose

        self.dataset_files = []
        self.dataset_len = -1
        self.prepare_dataset(dataset_dir)

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len

        if self.verbose:
            print('Dataset Info:')
            print('-- mode: ', self.mode)
            print('-- total frames: ', self.dataset_len)
            print('-- obs_len: ', self.obs_len)
            print('-- pred_len: ', self.pred_len)
            print('-- seq_len: ', self.seq_len)
            print('-- aug: ', self.aug)

    def prepare_dataset(self, feat_path):
        if self.verbose:
            print("preparing {}".format(feat_path))

        if isinstance(feat_path, list):
            for path in feat_path:
                sequences = os.listdir(path)
                for seq in sequences:
                    file_path = f"{path}/{seq}"
                    self.dataset_files.append(file_path)
        else:
            sequences = os.listdir(feat_path)
            for seq in sequences:
                file_path = f"{feat_path}/{seq}"
                self.dataset_files.append(file_path)

        self.dataset_len = len(self.dataset_files)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        df = pd.read_pickle(self.dataset_files[idx])
        '''
            "SEQ_ID", "CITY_NAME", "ORIG", "ROT", "TIMESTAMP", "TRAJS", "PAD_FLAGS", "MLG"
        '''

        data = self.data_augmentation(df)

        seq_id = data['SEQ_ID']
        city_name = data['CITY_NAME']
        orig = data['ORIG']
        rot = data['ROT']

        timestamp = data['TIMESTAMP']

        trajs = data['TRAJS']
        trajs_obs = trajs[:, :self.obs_len]
        trajs_fut = trajs[:, self.obs_len:]

        pad_flags = data['PAD_FLAGS']
        pad_obs = pad_flags[:, :self.obs_len]
        pad_fut = pad_flags[:, self.obs_len:]

        mlg = data['MLG']

        # Recover DA: add self-loop, add directed edge
        graph_da = mlg['DA']
        npts_da = len(graph_da['ctrs'])
        for i, edges in graph_da['ms_edges'].items():
            u = np.concatenate([np.arange(npts_da), edges['u'], edges['v']])
            v = np.concatenate([np.arange(npts_da), edges['v'], edges['u']])
            edges['u'] = u
            edges['v'] = v

        # Render S-T DA
        st_da = np.zeros((npts_da, self.obs_len))
        tree_da = spatial.KDTree(graph_da['ctrs'])
        for t in range(self.obs_len):
            pos = trajs_obs[:, t, :]
            pos = pos[pad_obs[:, t] == 0]
            # ~ a. nearest cell within a range
            # dist, idcs = tree_da.query(pos[:, :2])
            # idcs = idcs[dist < 1.0]
            # ~ b. cells within a range
            pair_list = tree_da.query_ball_point(pos[:, :2], r=1.0)
            idcs = []
            for pairs in pair_list:
                idcs += pairs
            st_da[idcs, t] = 1
        graph_da['feats'] = np.concatenate([graph_da['ctrs'], st_da], axis=1)  # (N, 22)

        if self.aug:
            # Re-calculate DA2LS since data augmentation
            graph_ls = mlg['LS']
            ctrs_ls = graph_ls['ctrs']
            pair_list = tree_da.query_ball_point(ctrs_ls, r=2.5)  # 2.25
            edges = []
            for i, pairs in enumerate(pair_list):
                if len(pairs):
                    tmp = np.stack([pairs, np.full(len(pairs), i)], axis=1)
                    edges.append(tmp)
            mlg['DA2LS'] = np.concatenate(edges)

        # Get the nearest point to GT goal in DA
        goal_gt = trajs[0, -1]

        sigma = 2.0
        dist2 = np.sum((graph_da['ctrs'] - goal_gt)**2, axis=1)
        goda_label = np.exp(-dist2 / (2 * sigma**2))  # gaussian kernel
        goda_idcs = np.where(goda_label > 0.85)[0]
        if not len(goda_idcs):
            _, goda_idx = tree_da.query(goal_gt)
            goda_idcs = np.array([goda_idx])
        # print('idcs: {}, dist: {}'.format(idcs, dist))

        data = {}
        data['SEQ_ID'] = seq_id
        data['CITY_NAME'] = city_name
        data['ORIG'] = orig
        data['ROT'] = rot
        data['DT'] = np.diff(timestamp, prepend=timestamp[0])[:self.obs_len]
        data['TRAJS_OBS'] = trajs_obs
        data['TRAJS_FUT'] = trajs_fut
        data['PAD_OBS'] = pad_obs
        data['PAD_FUT'] = pad_fut
        data['MLG_DA'] = mlg['DA']
        data['MLG_LS'] = mlg['LS']
        data['MLG_DA2LS'] = mlg['DA2LS']

        # ~ currently we only consider the target agent
        data['GOAL_GT'] = goal_gt  # (2,)
        data['GODA_IDCS'] = goda_idcs  # (x,)
        data['GODA_LABEL'] = goda_label

        # _, ax = plt.subplots(figsize=(12, 12))
        # ax.axis('equal')
        # cmap = cm.get_cmap('jet')
        # for t in range(self.obs_len):
        #     pts_t = graph_da['ctrs'][np.where(st_da[:, t])]
        #     ax.scatter(pts_t[:, 0], pts_t[:, 1], color=cmap(t/self.obs_len), marker='o', alpha=0.5)
        # plt.show()

        return data

    def collate_fn(self, batch):
        batch = from_numpy(batch)
        return_batch = dict()
        return_batch['BATCH_SIZE'] = len(batch)
        # Batching by use a list for non-fixed size
        for key in batch[0].keys():
            return_batch[key] = [x[key] for x in batch]
        return return_batch

    def data_augmentation(self, df):
        '''
            "SEQ_ID", "CITY_NAME", "ORIG", "ROT", "TIMESTAMP", "TRAJS", "PAD_FLAGS", "MLG"
        '''

        data = {}
        for key in list(df.keys()):
            data[key] = df[key].values[0]

        is_aug = random.choices([True, False], weights=[0.95, 0.05])[0]
        if not (self.aug and is_aug):
            return data

        trajs = data['TRAJS']  # (N_a, 50(20), 2)
        mlg = data['MLG']

        graph_da = mlg['DA']
        graph_ls = mlg['LS']

        # ~ random vertical flip
        if random.choice([True, False]):
            # trajs
            trajs[..., 1] *= -1.0
            # DA
            graph_da['ctrs'][..., 1] *= -1.0
            # LS
            graph_ls['ctrs'][..., 1] *= -1.0
            graph_ls['feats'][..., 1] *= -1.0

        # ~ random rotate
        theta = np.random.uniform(-np.pi/3, np.pi/3)
        rot_aug = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
        # traj
        trajs[..., 0:2] = trajs[..., 0:2].dot(rot_aug)
        # DA
        graph_da['ctrs'][..., 0:2] = graph_da['ctrs'][..., 0:2].dot(rot_aug)
        # LS
        graph_ls['ctrs'][..., 0:2] = graph_ls['ctrs'][..., 0:2].dot(rot_aug)
        graph_ls['feats'][..., 0:2] = graph_ls['feats'][..., 0:2].dot(rot_aug)

        # ~ random perturbation
        # trajs
        trajs += np.random.uniform(-0.05, 0.05, trajs.shape)
        # DA
        graph_da['ctrs'] += np.random.uniform(-0.1, 0.1, graph_da['ctrs'].shape)

        data['TRAJS'] = trajs
        data['MLG']['DA'] = graph_da
        data['MLG']['LS'] = graph_ls

        # ~ random drop

        return data
