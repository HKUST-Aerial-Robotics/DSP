import os
import sys
import time
from typing import Any, Dict, List, Tuple, Union
from datetime import datetime
import argparse
import numpy as np
import faulthandler
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from loader import Loader
from utils.utils import AverageMeter, AverageMeterForDict, str2bool


def parse_arguments() -> Any:
    """Arguments for running the baseline.

    Returns:
        parsed arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode",
                        default="train",
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
    parser.add_argument("--val_batch_size",
                        type=int,
                        default=512,
                        help="Val batch size")
    parser.add_argument("--model",
                        default="dsp",
                        type=str,
                        help="Name of model")
    parser.add_argument("--loss",
                        default="dsp",
                        type=str,
                        help="Type of loss function")
    parser.add_argument("--data_aug",
                        action="store_true",
                        help="Enable data augmentation")
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

    return parser.parse_args()


def main():
    args = parse_arguments()
    print('Args: ', args)

    faulthandler.enable()

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device('cpu')

    if not args.model_path.endswith(".tar"):
        assert False, "Model path error - '{}'".format(args.model_path)

    loader = Loader(args, device, is_ddp=False)
    print('[Resume]Loading state_dict from {}'.format(args.model_path))
    loader.set_resmue(args.model_path)
    (train_set, val_set), net, _, _, evaluator = loader.load()

    dl_val = DataLoader(val_set,
                        batch_size=args.val_batch_size,
                        shuffle=False,
                        num_workers=8,
                        collate_fn=val_set.collate_fn,
                        drop_last=False,
                        pin_memory=True)

    net.eval()

    with torch.no_grad():
        # * Validation
        val_start = time.time()
        val_eval_meter = AverageMeterForDict()
        for i, data in enumerate(tqdm(dl_val)):
            out = net(data)
            post_out = net.post_process(out)

            eval_out = evaluator.evaluate(post_out, data)
            val_eval_meter.update(eval_out, n=data['BATCH_SIZE'])

        print('\nValidation set finish, cost {} secs'.format(time.time() - val_start))
        print('-- ' + val_eval_meter.get_info())

    print('\nExit...')


if __name__ == "__main__":
    main()
