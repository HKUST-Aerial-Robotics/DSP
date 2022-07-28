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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from loader import Loader
from utils.logger import Logger
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
    parser.add_argument("--local_rank", default=-1)  # for DDP

    return parser.parse_args()


def distributed_concat(tensor):
    output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat


def main():
    args = parse_arguments()

    local_rank = int(args.local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    device = torch.device("cuda", local_rank)
    world_size = dist.get_world_size()

    is_main = True if local_rank == 0 else False

    # logger only for print
    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "log/" + date_str
    logger = Logger(date_str=date_str, enable=is_main, log_dir=log_dir,
                    enable_flags={'writer': False, 'mailbot': False})


    loader = Loader(args, device, is_ddp=True, world_size=world_size, local_rank=local_rank)
    logger.print('[Resume]Loading state_dict from {}'.format(args.model_path))
    loader.set_resmue(args.model_path)
    test_set, net, _, _, _ = loader.load()

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
    dataloader = DataLoader(test_set,
                            batch_size=32,
                            num_workers=32,
                            drop_last=False,
                            collate_fn=test_set.collate_fn,
                            sampler=test_sampler,
                            pin_memory=True)

    net.eval()

    # data needed:
    # post_out: traj, prob
    # data: seq_id, orig, rot
    seq_id_vec = []
    orig_vec = []
    rot_vec = []
    traj_vec = []
    prob_vec = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, disable=(not is_main))):
            out = net(data)
            post_out = net.module.post_process(out)

            # List[int], batch
            # List[Tensor], batch, (2,)
            # List[Tensor], batch, (2, 2)
            seq_id = torch.Tensor(data['SEQ_ID']).to(device)  # batch
            orig = torch.stack(data['ORIG']).to(device)  # (batch, 2)
            rot = torch.stack(data['ROT']).to(device)  # (batch, 2, 2)

            seq_id_vec.append(seq_id)
            orig_vec.append(orig)
            rot_vec.append(rot)

            traj_pred = post_out['traj_pred']  # batch x n_mode x 30 x 2
            traj_prob = post_out['prob_pred']  # batch x n_mode

            traj_vec.append(traj_pred)
            prob_vec.append(traj_prob)

        seq_id_vec = torch.cat(seq_id_vec)
        orig_vec = torch.cat(orig_vec)
        rot_vec = torch.cat(rot_vec)
        traj_vec = torch.cat(traj_vec)
        prob_vec = torch.cat(prob_vec)

        seq_id_vec = distributed_concat(seq_id_vec)
        orig_vec = distributed_concat(orig_vec)
        rot_vec = distributed_concat(rot_vec)
        traj_vec = distributed_concat(traj_vec)
        prob_vec = distributed_concat(prob_vec)

        logger.print('gathering results:')
        logger.print('seq_id_vec: {}'.format(seq_id_vec.shape))
        logger.print('orig_vec: {}'.format(orig_vec.shape))
        logger.print('rot_vec: {}'.format(rot_vec.shape))
        logger.print('traj_vec: {}'.format(traj_vec.shape))
        logger.print('prob_vec: {}'.format(prob_vec.shape))

    if is_main:
        submitters = import_module('{}'.format(args.submitter)).Submitter()

        seq_id_vec = seq_id_vec.cpu().detach().numpy()
        orig_vec = orig_vec.cpu().detach().numpy()
        rot_vec = rot_vec.cpu().detach().numpy()
        traj_vec = traj_vec.cpu().detach().numpy()
        prob_vec = prob_vec.cpu().detach().numpy()

        res = (seq_id_vec, orig_vec, rot_vec, traj_vec, prob_vec)
        submitters.format_and_submit(res)

    dist.barrier()  # sync

    dist.destroy_process_group()
    logger.print('\nExit...')


if __name__ == "__main__":
    main()
