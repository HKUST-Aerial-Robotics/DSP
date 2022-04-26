import os
import numpy as np
import random

import torch
import torch.nn.functional as F


def cls_nll_loss(pred, gt_label, device):
    '''
        pred: List of Tensor, length = batch
        gt_label: np.ndarray, batch x 1
    '''
    batch_size = len(pred)
    loss_batch = torch.zeros(batch_size).to(device)

    for idx in range(batch_size):
        pred_enc = pred[idx]
        gt_enc = F.one_hot(gt_label[idx], pred_enc.shape[0]).float()

        loss = F.binary_cross_entropy(pred_enc, gt_enc)
        loss_batch[idx] = loss

    return torch.mean(loss_batch)


def multicls_nll_loss(pred, gt_labels, device):
    '''
        pred: List of Tensor, length = batch
        gt_labels: List[List[int]], length = batch
    '''
    assert len(gt_labels) == len(pred), 'Batch size not match'
    batch_size = len(pred)
    loss_batch = torch.zeros(batch_size).to(device)

    for idx in range(batch_size):
        pred_enc = pred[idx]

        labels = torch.tensor(gt_labels[idx]).to(device)
        gt_enc = torch.zeros(pred_enc.shape[0]).to(device).scatter_(0, labels, 1.0)

        loss = F.binary_cross_entropy(pred_enc, gt_enc)
        loss_batch[idx] = loss

    return torch.mean(loss_batch)


def reg_gmm_nll_loss(alpha, pred, gt):
    muX = pred[..., 0]
    muY = pred[..., 1]
    sigX = pred[..., 2]  # ! reciprocals
    sigY = pred[..., 3]  # ! reciprocals
    rho = pred[..., 4]

    x = gt[..., 0]
    y = gt[..., 1]

    ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)

    a = ohr * sigX * sigY / (2 * np.pi)
    b = -0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2)
                                    * torch.pow(y - muY, 2) - 2 * rho * sigX * sigY * (x - muX) * (y - muY))

    w = torch.stack([alpha for _ in range(x.shape[-1])], dim=2)
    L = torch.sum(a * torch.exp(b) * w, dim=1) + 1e-6

    out = -torch.log(L)

    return torch.mean(out, dim=1)


def reg_nll_loss(traj_pred, traj_gt):
    muX = traj_pred[..., 0]
    muY = traj_pred[..., 1]
    sigX = traj_pred[..., 2]  # ! reciprocals
    sigY = traj_pred[..., 3]  # ! reciprocals
    rho = traj_pred[..., 4]
    ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
    x = traj_gt[..., 0]
    y = traj_gt[..., 1]

    '''
        Here we have two methods calculating negative log-likelihood:
            1. Use original gaussian equation
            2. Simplify the formulation above
        Mathematically, these two methods generate same results. However, since sometimes the 
        likelihood L can be extremely small, leading to a NaN issue. If we add a small number (1e-6),
        the results of the two methods will be different. Surprisingly, the simplified formulation
        converges slower than the original formulation in practice, despite the simplified version 
        does not suffer from the numerical issue. The reason may be the probability density of the
        point that far away from the mean value is too small, leading to a unstable gradient. 

        (Gradient clipping can also improve the performance.)
    '''

    # Original gaussian distribution
    # a = ohr * sigX * sigY / (2 * np.pi)
    # b = -0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2)
    #                                 * torch.pow(y - muY, 2) - 2 * rho * sigX * sigY * (x - muX) * (y - muY))
    # L = a * torch.exp(b) + 1e-6  # constant for numerical issue. it can also provide stable gradient
    # out = -torch.log(L)

    # Simplified NLL
    out = 0.5 * torch.pow(ohr, 2) * (torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(
        y - muY, 2) - 2 * rho * sigX * sigY * (x - muX) * (y - muY)) - torch.log(sigX * sigY * ohr) + np.log(2*np.pi)

    return torch.mean(out, dim=-1)


def reg_mse_loss(traj_pred, traj_gt):
    X = traj_pred[..., 0]
    Y = traj_pred[..., 1]
    x = traj_gt[..., 0]
    y = traj_gt[..., 1]
    out = torch.pow(x - X, 2) + torch.pow(y - Y, 2)

    return torch.mean(out, dim=-1)


def reg_rmse_loss(traj_pred, traj_gt):
    X = traj_pred[..., 0]
    Y = traj_pred[..., 1]
    x = traj_gt[..., 0]
    y = traj_gt[..., 1]
    out = torch.sqrt(torch.pow(x - X, 2) + torch.pow(y - Y, 2))

    return torch.mean(out, dim=-1)


def reg_huber_loss(pred, gt):
    X = pred[..., 0]
    Y = pred[..., 1]
    x = gt[..., 0]
    y = gt[..., 1]

    rmse = torch.sqrt(torch.pow(x - X, 2) + torch.pow(y - Y, 2))
    out = F.smooth_l1_loss(rmse, torch.zeros_like(rmse), reduction='none')
    return out


def reduce_loss(loss, reduction):
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    if weight is not None:
        loss = loss * weight

    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def focal_bce_loss(pred, target, weight=None, gamma=2.0, alpha=0.9, reduction='mean', avg_factor=None):
    '''
        pred: [0, 1]
    '''
    pt = (1 - pred) * target + pred * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy(pred, target, reduction='none') * focal_weight

    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def variant_focal_bce_loss(pred, target, alpha=2.0, beta=4.0, thres=1.0, sigma=0.5):
    '''
        pred: [0, 1]
    '''
    pos_inds = target.ge(thres).float()
    neg_inds = target.lt(thres).float()

    neg_weights = torch.pow(1 - target, beta)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    loss = 0
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss * sigma + neg_loss * (1 - sigma)) / num_pos

    return loss
