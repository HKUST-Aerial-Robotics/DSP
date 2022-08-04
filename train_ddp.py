import os
import sys
import time
import copy
from typing import Any, Dict, List, Tuple, Union
from datetime import datetime
import argparse
import faulthandler
from tqdm import tqdm
import numpy as np
#
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
#
from loader import Loader
from utils.logger import Logger
from utils.utils import AverageMeter, AverageMeterForDict, check_loss_abnormal, str2bool
from utils.utils import save_ckpt


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
                        default=16,
                        help="Training batch size")
    parser.add_argument("--val_batch_size",
                        type=int,
                        default=16,
                        help="Val batch size")
    parser.add_argument("--train_epoches",
                        type=int,
                        default=10,
                        help="Number of epoches for training")
    parser.add_argument("--val_interval",
                        type=int,
                        default=5,
                        help="Validation intervals")
    parser.add_argument("--model",
                        default="dsp",
                        type=str,
                        help="Name of model")
    parser.add_argument("--loss",
                        default="dsp",
                        type=str,
                        help="Type of loss function")
    parser.add_argument("--saved_model_dir",
                        required=False,
                        type=str,
                        help="path to the saved model")
    parser.add_argument("--data_aug",
                        action="store_true",
                        help="Enable data augmentation")
    parser.add_argument("--use_cuda",
                        type=bool,
                        default=True,
                        help="Use CUDA for acceleration")
    parser.add_argument("--logger_writer",
                        type=str2bool,
                        default=True,
                        help="Enable tensorboard")
    parser.add_argument("--logger_mailbot",
                        type=str2bool,
                        default=False,
                        help="Enable mailbot")
    parser.add_argument("--adv_cfg_path",
                        required=True,
                        default="",
                        type=str)
    parser.add_argument("--resume",
                        action="store_true",
                        help="Resume training")
    parser.add_argument("--model_path",
                        required=False,
                        type=str,
                        help="path to the saved model")
    parser.add_argument("--local_rank", default=-1)  # for DDP

    return parser.parse_args()


def distributed_mean(tensor):
    output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.stack(output_tensors, dim=0)
    return concat.mean(0)


def main():
    args = parse_arguments()
    faulthandler.enable()
    start_time = time.time()

    local_rank = int(args.local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    device = torch.device("cuda", local_rank)
    world_size = dist.get_world_size()

    is_main = True if local_rank == 0 else False

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "log/" + date_str
    logger = Logger(date_str=date_str, enable=is_main, log_dir=log_dir,
                    enable_flags={'writer': args.logger_writer, 'mailbot': args.logger_mailbot})

    loader = Loader(args, device, is_ddp=True, world_size=world_size, local_rank=local_rank)
    if args.resume:
        logger.print('[Resume]Loading state_dict from {}'.format(args.model_path))
        loader.set_resmue(args.model_path)
    (train_set, val_set), net, loss_fn, optimizer, evaluator = loader.load()

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    dl_train = DataLoader(train_set,
                          batch_size=args.train_batch_size,
                          num_workers=48,
                          collate_fn=train_set.collate_fn,
                          drop_last=True,
                          sampler=train_sampler,
                          pin_memory=True)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
    dl_val = DataLoader(val_set,
                        batch_size=args.val_batch_size,
                        num_workers=48,
                        collate_fn=val_set.collate_fn,
                        drop_last=True,
                        sampler=val_sampler,
                        pin_memory=True)

    niter = 0
    best_metric = 10e2
    metric_name = 'brier_fde_k'

    train_loss_hist = []
    for epoch in range(args.train_epoches):
        dist.barrier()  # sync
        logger.print('\nEpoch {}'.format(epoch))

        # * Train
        dl_train.sampler.set_epoch(epoch)
        epoch_start = time.time()
        train_loss_meter = AverageMeterForDict()
        train_eval_meter = AverageMeterForDict()
        net.train()
        for i, data in enumerate(tqdm(dl_train, disable=(not is_main))):
            out = net(data)
            loss_out = loss_fn(out, data)

            post_out = net.module.post_process(out)
            eval_out = evaluator.evaluate(post_out, data)

            optimizer.zero_grad()
            loss_out['loss'].backward()
            lr = optimizer.step(epoch)

            train_loss_meter.update(loss_out)
            train_eval_meter.update(eval_out)
            niter += world_size * args.train_batch_size
            logger.add_dict(loss_out, niter, prefix='train/')

        loss_avg = train_loss_meter.metrics['loss'].avg
        logger.print('[Training] Avg. loss: {:.6}, time cost: {:.3} mins, lr: {:.3}'.
                     format(loss_avg, (time.time() - epoch_start) / 60.0, lr))
        logger.print('-- ' + train_eval_meter.get_info())

        for key, elem in train_eval_meter.metrics.items():
            logger.add_scalar(title='train/{}'.format(key), value=elem.avg, it=epoch)

        dist.barrier()  # sync
        if (epoch + 1) % args.val_interval == 0:
            # * Validation
            with torch.no_grad():
                val_start = time.time()
                dl_val.sampler.set_epoch(epoch)
                val_loss_meter = AverageMeterForDict()
                val_eval_meter = AverageMeterForDict()
                net.eval()
                for i, data in enumerate(tqdm(dl_val, disable=(not is_main))):
                    out = net(data)
                    loss_out = loss_fn(out, data)

                    post_out = net.module.post_process(out)
                    eval_out = evaluator.evaluate(post_out, data)

                    val_loss_meter.update(loss_out)
                    val_eval_meter.update(eval_out)

                # make eval results into a Tensor
                eval_res = [elem.avg for k, elem in val_eval_meter.metrics.items()]
                eval_res = torch.from_numpy(np.array(eval_res)).to(device)

                val_eval_mean = distributed_mean(eval_res)
                val_eval = dict()
                for i, key in enumerate(list(val_eval_meter.metrics.keys())):
                    val_eval[key] = val_eval_mean[i].item()
                val_eval_meter.reset()
                val_eval_meter.update(val_eval)

                logger.print('[Validation] Avg. loss: {:.6}, time cost: {:.3} mins'.format(
                    val_loss_meter.metrics['loss'].avg, (time.time() - val_start) / 60.0))
                logger.print('-- ' + val_eval_meter.get_info())

                for key, elem in val_loss_meter.metrics.items():
                    logger.add_scalar(title='val/{}'.format(key), value=elem.avg, it=epoch)
                for key, elem in val_eval_meter.metrics.items():
                    logger.add_scalar(title='val/{}'.format(key), value=elem.avg, it=epoch)

                if is_main and (epoch >= args.train_epoches / 2):
                    if val_eval_meter.metrics[metric_name].avg < best_metric:
                        model_name = date_str + '_{}_ddp_best.tar'.format(args.model)
                        save_ckpt(net.module, optimizer, epoch, 'saved_models/', model_name)
                        best_metric = val_eval_meter.metrics[metric_name].avg
                        logger.print('Save the model: {}, {}: {:.4}, epoch: {}'.format(model_name, metric_name, best_metric, epoch))

        if is_main:
            if int(100 * epoch / args.train_epoches) in [20, 40, 60, 80]:
                model_name = date_str + '_{}_ddp_ckpt_epoch{}.tar'.format(args.model, epoch)               
                save_ckpt(net.module, optimizer, epoch, 'saved_models/', model_name)
                logger.print('Save the model to {}'.format('saved_models/' + model_name))

    logger.print("\nTraining completed in {} mins".format((time.time() - start_time) / 60.0))

    if is_main:
        # save trained model
        model_name = date_str + '_{}_ddp_epoch{}.tar'.format(args.model, args.train_epoches)
        save_ckpt(net.module, optimizer, epoch, 'saved_models/', model_name)
        logger.print('Save the model to {}'.format('saved_models/' + model_name))

    dist.destroy_process_group()
    logger.print('\nExit...')


if __name__ == "__main__":
    main()
