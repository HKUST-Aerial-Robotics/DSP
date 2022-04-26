import os
import sys
import time
from typing import Any, Dict, List, Tuple, Union
from datetime import datetime
import argparse
import faulthandler
from tqdm import tqdm
#
import numpy as np
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader
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
                        help="path to the file which has features.")
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
    parser.add_argument("--clipping",
                        default=1.0,
                        type=float,
                        help="Max norm of gradient clipping. 0 for disable gradient clipping")
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

    return parser.parse_args()


def main():
    args = parse_arguments()
    print('Args: ', args)

    faulthandler.enable()
    start_time = time.time()

    # torch.multiprocessing.set_sharing_strategy('file_system')

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device('cpu')

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "log/" + date_str
    logger = Logger(date_str=date_str, log_dir=log_dir,
                    enable_flags={'writer': args.logger_writer})

    loader = Loader(args, device, is_ddp=False)
    if args.resume:
        logger.print('[Resume]Loading state_dict from {}'.format(args.model_path))
        loader.set_resmue(args.model_path)
    (train_set, val_set), net, loss_fn, optimizer, evaluator = loader.load()

    dl_train = DataLoader(train_set,
                          batch_size=args.train_batch_size,
                          shuffle=True,
                          num_workers=8,
                          collate_fn=train_set.collate_fn,
                          drop_last=True,
                          pin_memory=True)
    dl_val = DataLoader(val_set,
                        batch_size=args.val_batch_size,
                        shuffle=False,
                        num_workers=8,
                        collate_fn=val_set.collate_fn,
                        drop_last=True,
                        pin_memory=True)

    niter = 0
    best_metric_val = 10e2
    flag_metric = 'brier_fde_k'

    for epoch in range(args.train_epoches):
        logger.print('\nEpoch {}'.format(epoch))

        # * Train
        epoch_start = time.time()
        train_loss_meter = AverageMeterForDict()
        train_eval_meter = AverageMeterForDict()
        net.train()
        for i, data in enumerate(tqdm(dl_train)):
            out = net(data)
            loss_out = loss_fn(out, data)

            post_out = net.post_process(out)
            eval_out = evaluator.evaluate(post_out, data)

            optimizer.zero_grad()
            loss_out['loss'].backward()
            if bool(args.clipping):
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clipping)
            lr = optimizer.step(epoch)

            train_loss_meter.update(loss_out)
            train_eval_meter.update(eval_out)
            niter += args.train_batch_size
            logger.add_dict(loss_out, niter, prefix='train/')

        loss_avg = train_loss_meter.metrics['loss'].avg

        logger.print('[Training] Avg. loss: {:.6}, time cost: {:.3} mins, lr: {:.3}'.
                     format(loss_avg, (time.time() - epoch_start) / 60.0, lr))
        logger.print('-- ' + train_eval_meter.get_info())

        for key, elem in train_eval_meter.metrics.items():
            logger.add_scalar(title='train/{}'.format(key), value=elem.avg, it=epoch)

        if (epoch + 1) % args.val_interval == 0:
            # * Validation
            with torch.no_grad():
                val_start = time.time()
                val_loss_meter = AverageMeterForDict()
                val_eval_meter = AverageMeterForDict()
                net.eval()
                for i, data in enumerate(tqdm(dl_val)):
                    out = net(data)
                    loss_out = loss_fn(out, data)

                    post_out = net.post_process(out)
                    eval_out = evaluator.evaluate(post_out, data)

                    val_loss_meter.update(loss_out)
                    val_eval_meter.update(eval_out)

                logger.print('[Validation] Avg. loss: {:.6}, time cost: {:.3} mins'.format(
                    val_loss_meter.metrics['loss'].avg, (time.time() - val_start) / 60.0))
                logger.print('-- ' + val_eval_meter.get_info())

                for key, elem in val_loss_meter.metrics.items():
                    logger.add_scalar(title='val/{}'.format(key), value=elem.avg, it=epoch)
                for key, elem in val_eval_meter.metrics.items():
                    logger.add_scalar(title='val/{}'.format(key), value=elem.avg, it=epoch)

                if (epoch >= args.train_epoches / 2):
                    if val_eval_meter.metrics[flag_metric].avg < best_metric_val:
                        model_name = date_str + '_{}_best.tar'.format(args.model)
                        save_ckpt(net, optimizer, epoch, 'saved_models/', model_name)
                        best_metric_val = val_eval_meter.metrics[flag_metric].avg
                        print('Save the model: {}, {}: {:.4}, epoch: {}'.format(
                            model_name, flag_metric, best_metric_val, epoch))

    logger.print("\nTraining completed in {} mins".format((time.time() - start_time) / 60.0))

    # save trained model
    model_name = date_str + '_{}_epoch{}.tar'.format(args.model, args.train_epoches)
    save_ckpt(net, optimizer, epoch, 'saved_models/', model_name)
    print('Save the model to {}'.format('saved_models/' + model_name))

    print('\nExit...')


if __name__ == "__main__":
    main()
