import os
import sys
from datetime import datetime
#
from argoverse.evaluation.competition_util import generate_forecasting_h5


class Submitter():
    def __init__(self):
        print('[Submitter] Argo dataset and DSP')
        date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.filename = date_str + '_dsp_res'

    def format_and_submit(self, out_vec, data_vec):
        assert len(out_vec) == len(data_vec), 'Wrong size for out and data'
        len_vec = len(out_vec)

        output_all = {}
        probability_all = {}

        for i in range(len_vec):
            data = data_vec[i]
            out = out_vec[i]

            seq_id_batch = data['SEQ_ID']  # batch
            origin_batch = data['ORIG']  # batch x 2
            rot_mat_batch = data['ROT']  # batch x 2 x 2

            traj_pred_batch = out['traj_pred']
            traj_prob_batch = out['prob_pred']
            # traj_pred_batch:    batch x n_mod x pred_len x 5
            # traj_prob_batch:    batch x n_mod

            batch_size = len(seq_id_batch)
            for j in range(batch_size):
                seq_id = seq_id_batch[j]
                rot = rot_mat_batch[j].cpu().detach().numpy()
                orig = origin_batch[j].cpu().detach().numpy()

                traj_pred = traj_pred_batch[j].cpu().detach().numpy()[..., :2]  # n_mod x seq x 2
                traj_prob = traj_prob_batch[j].cpu().detach().numpy()

                output_all[seq_id] = traj_pred.dot(rot.T) + orig
                probability_all[seq_id] = traj_prob

        # output
        output_path = 'competition_files/'
        generate_forecasting_h5(output_all, output_path, probabilities=probability_all, filename=self.filename)
