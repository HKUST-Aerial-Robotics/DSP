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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from loader import Loader
from utils.logger import Logger
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
                        help="path to the file which has features."
                        )
    parser.add_argument("--obs_len",
                        default=20,
                        type=int,
                        help="Observed length of the trajectory")
    parser.add_argument("--pred_len",
                        default=30,
                        type=int,
                        help="Prediction Horizon")
    parser.add_argument("--train_batch_size",
                        type=int,
                        default=512,
                        help="Training batch size")
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
    parser.add_argument("--local_rank", default=-1)  # for DDP

    return parser.parse_args()


def distributed_concat(tensor):
    output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat


def distributed_mean(tensor):
    output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.stack(output_tensors, dim=0)
    return concat.mean(0)


def main():
    args = parse_arguments()
    faulthandler.enable()

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
    (_, val_set), net, _, _, evaluator = loader.load()

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
    dl_val = DataLoader(val_set,
                        batch_size=args.val_batch_size,
                        num_workers=16,
                        collate_fn=val_set.collate_fn,
                        drop_last=False,
                        sampler=val_sampler,
                        pin_memory=True)

    net.eval()
    with torch.no_grad():
        # * Validation
        dl_val.sampler.set_epoch(0)
        val_start = time.time()
        val_eval_meter = AverageMeterForDict()
        for i, data in enumerate(tqdm(dl_val, disable=(not is_main))):
            out = net(data)
            post_out = net.module.post_process(out)

            eval_out = evaluator.evaluate(post_out, data)
            val_eval_meter.update(eval_out, n=data['BATCH_SIZE'])

        # make eval results into a Tensor
        eval_res = [elem.avg for k, elem in val_eval_meter.metrics.items()]
        eval_res = torch.from_numpy(np.array(eval_res)).to(device)

        val_eval_mean = distributed_mean(eval_res)
        val_eval = dict()
        for i, key in enumerate(list(val_eval_meter.metrics.keys())):
            val_eval[key] = val_eval_mean[i].item()
        val_eval_meter.reset()
        val_eval_meter.update(val_eval)

        dist.barrier()  # sync
        logger.print('\nValidation set finish, cost {} secs'.format(time.time() - val_start))
        logger.print('-- ' + val_eval_meter.get_info())


    dist.destroy_process_group()
    logger.print('\nExit...')


if __name__ == "__main__":
    main()
