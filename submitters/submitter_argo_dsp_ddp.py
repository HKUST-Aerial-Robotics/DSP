import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union
#
import numpy as np
import torch
#
from argoverse.evaluation.competition_util import generate_forecasting_h5


class Submitter():
    def __init__(self):
        print('[Submitter] Argo dataset and DSP')
        date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.filename = date_str + '_dsp_res'

    def format_and_submit(self, res):
        seq_id_vec, orig_vec, rot_vec, traj_vec, prob_vec = res

        output_all = {}
        probability_all = {}
        for seq_id, orig, rot, traj, prob in zip(seq_id_vec, orig_vec, rot_vec, traj_vec, prob_vec):
            traj = traj[..., :2]  # n_mod x seq x 2
            output_all[seq_id] = traj.dot(rot.T) + orig
            probability_all[seq_id] = prob

        print('[Submitter] Saving {} results'.format(len(output_all)))
        # output
        output_path = 'competition_files/'
        generate_forecasting_h5(output_all, output_path, probabilities=probability_all, filename=self.filename)
