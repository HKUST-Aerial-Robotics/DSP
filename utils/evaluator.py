import math
import os
import numpy as np
import torch
from argoverse.evaluation.eval_forecasting import get_displacement_errors_and_miss_rate


class TrajPredictionEvaluator():
    ''' Return evaluation results for batched data '''

    def __init__(self):
        super(TrajPredictionEvaluator, self).__init__()
        self.miss_thres = 2.0

    def evaluate(self, post_out, data):
        traj_pred = post_out['traj_pred']
        prob_pred = post_out['prob_pred']
        # traj_pred:    batch x n_mod x pred_len x 5
        # prob_pred:    batch x n_mod

        traj_fut = torch.stack([traj[0, :, 0:2] for traj in data['TRAJS_FUT']])  # batch x fut x 2
        n_mod = traj_pred.shape[1]

        # to np.ndarray
        traj_pred = np.asarray(traj_pred.cpu().detach().numpy()[:, :, :, :2], np.float32)
        prob_pred = np.asarray(prob_pred.cpu().detach().numpy(), np.float32)
        traj_fut = np.asarray(traj_fut.numpy(), np.float32)

        seq_id_batch = data['SEQ_ID']
        batch_size = len(seq_id_batch)

        pred_dict = {}
        gt_dict = {}
        prob_dict = {}
        for j in range(batch_size):
            seq_id = seq_id_batch[j]
            pred_dict[seq_id] = traj_pred[j]
            gt_dict[seq_id] = traj_fut[j]
            prob_dict[seq_id] = prob_pred[j]

        # # Max #guesses (K): 1
        res_1 = get_displacement_errors_and_miss_rate(
            pred_dict, gt_dict, 1, 30, miss_threshold=self.miss_thres, forecasted_probabilities=prob_dict)
        # # Max #guesses (K): 6
        res_k = get_displacement_errors_and_miss_rate(
            pred_dict, gt_dict, 6, 30, miss_threshold=self.miss_thres, forecasted_probabilities=prob_dict)

        eval_out = {}
        eval_out['minade_1'] = res_1['minADE']
        eval_out['minfde_1'] = res_1['minFDE']
        eval_out['mr_1'] = res_1['MR']
        eval_out['brier_fde_1'] = res_1['brier-minFDE']

        eval_out['minade_k'] = res_k['minADE']
        eval_out['minfde_k'] = res_k['minFDE']
        eval_out['mr_k'] = res_k['MR']
        eval_out['brier_fde_k'] = res_k['brier-minFDE']

        return eval_out
