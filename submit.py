import os
import sys
import time
from typing import Any, Dict, List, Tuple, Union
from datetime import datetime
import argparse
import copy
import numpy as np
from importlib import import_module
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from loader import Loader
from utils.utils import AverageMeterForDict


def parse_arguments() -> Any:
    """Arguments for running the baseline.

    Returns:
        parsed arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode",
                        default="test",
                        type=str,
                        help="Mode, train/val/test")
    parser.add_argument("--features_dir",
                        required=True,
                        default="",
                        type=str,
                        help="path to the file which has features.")
    parser.add_argument("--obs_len",
                        default=20,
                        type=int,
                        help="Observed length of the trajectory")
    parser.add_argument("--pred_len",
                        default=30,
                        type=int,
                        help="Prediction Horizon")
    parser.add_argument("--model",
                        default="dsp",
                        type=str,
                        help="Name of model")
    parser.add_argument("--loss",
                        default="dsp",
                        type=str,
                        help="Type of loss function")
    parser.add_argument("--use_cuda",
                        type=bool,
                        default=True,
                        help="Use CUDA for acceleration")
    parser.add_argument("--adv_cfg_path",
                        required=True,
                        default="",
                        type=str)
    parser.add_argument("--model_path",
                        required=True,
                        type=str,
                        help="path to the saved model")
    parser.add_argument("--submitter",
                        default="argo_dsp",
                        type=str,
                        help="Name of submitters")

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device('cpu')

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    loader = Loader(args, device, is_ddp=False)

    print('[Resume]Loading state_dict from {}'.format(args.model_path))
    loader.set_resmue(args.model_path)
    test_set, net, _, _, _ = loader.load()
    print('Test set size: {}'.format(len(test_set)))

    net.eval()

    submitters = import_module('{}'.format(args.submitter)).Submitter()

    dataloader = DataLoader(test_set,
                            batch_size=64,
                            num_workers=8,
                            shuffle=False,
                            collate_fn=test_set.collate_fn,
                            pin_memory=True)

    out_vec = []
    data_vec = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            out = net(data)
            post_out = net.post_process(out)
            out_vec.append(post_out)

            meta = dict()
            meta['SEQ_ID'] = copy.deepcopy(data['SEQ_ID'])  # batch
            meta['ORIG'] = copy.deepcopy(data['ORIG'])  # batch x 2
            meta['ROT'] = copy.deepcopy(data['ROT'])  # batch x 2 x 2
            data_vec.append(meta)

    submitters.format_and_submit(out_vec, data_vec)

    print('\nExit...')


if __name__ == "__main__":
    main()
