import os
import sys
import time
from typing import Any, Dict, List, Tuple, Union
from datetime import datetime
import argparse
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from loader import Loader
from utils.logger import Logger
from utils.utils import AverageMeterForDict
from visualizers.visualizer_dsp import VisualizerDsp


def parse_arguments() -> Any:
    """Arguments for running the baseline.

    Returns:
        parsed arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode",
                        default="val",
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
    parser.add_argument("--shuffle",
                        action="store_true",
                        help="Shuffle order")
    parser.add_argument("--show_conditioned",
                        action="store_true",
                        help="Show missed sample only")
    return parser.parse_args()


def main():
    args = parse_arguments()
    print('Args: ', args)

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device('cpu')

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.model == 'dsp':
        vis = VisualizerDsp()
    else:
        assert False, "Unknown visualizer"

    if not args.model_path.endswith(".tar"):
        assert False, "Model path error - '{}'".format(args.model_path)

    if args.mode != 'test':
        loader = Loader(args, device, is_ddp=False)
        print('[Resume]Loading state_dict from {}'.format(args.model_path))
        loader.set_resmue(args.model_path)
        (train_set, val_set), net, _, _, _ = loader.load()
        net.eval()

        dl_train = DataLoader(train_set,
                              batch_size=1,
                              shuffle=args.shuffle,
                              num_workers=0,
                              collate_fn=train_set.collate_fn,
                              drop_last=False)
        dl_val = DataLoader(val_set,
                            batch_size=1,
                            shuffle=args.shuffle,
                            num_workers=0,
                            collate_fn=val_set.collate_fn,
                            drop_last=False)

        with torch.no_grad():
            for i, data in enumerate(tqdm(dl_val)):
                out = net(data)
                post_out = net.post_process(out)
                vis.draw_once(post_out, data, show_map=True)

    else:
        # test
        loader = Loader(args, device, is_ddp=False)
        print('[Resume]Loading state_dict from {}'.format(args.model_path))
        loader.set_resmue(args.model_path)
        test_set, net, _, _, _ = loader.load()
        net.eval()

        dl_test = DataLoader(test_set,
                             batch_size=1,
                             num_workers=0,
                             shuffle=False,
                             collate_fn=test_set.collate_fn)

        with torch.no_grad():
            for i, data in enumerate(tqdm(dl_test)):
                out = net(data)
                post_out = net.post_process(out)
                vis.draw_once(post_out, data, show_map=True, test_mode=True)

    print('\nExit...')


if __name__ == "__main__":
    main()
