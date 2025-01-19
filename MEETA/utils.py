import torch
import numpy as np
import random

import argparse
import torch.optim as optim

from collections import OrderedDict
import time
from torch.nn.utils.rnn import pad_sequence

import os

from sklearn import metrics

def get_aupr_davis(Y, P):
    Y = np.where(Y >= 7, 1, 0)
    P = np.where(P >= 7, 1, 0)
    #print(Y,P)
    prec, re, _ = metrics.precision_recall_curve(Y, P)
    aupr = metrics.auc(re, prec)
    return aupr

def get_aupr_kiba(Y, P):
    Y = np.where(Y >= 12.1, 1, 0)
    P = np.where(P >= 12.1, 1, 0)
    #print(Y,P)
    prec, re, _ = metrics.precision_recall_curve(Y, P)
    aupr = metrics.auc(re, prec)
    return aupr


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace



########## seed init ##########
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def pad_tensor_list(tensor_list):
    """
    Merge a list of tensors with shape (token, channel) into a single tensor
    with shape (batch, max_token, channel), and generate a mask tensor
    with shape (batch, max_token) where valid positions are 1 and padded positions are 0.

    Args:
        tensor_list (list of torch.Tensor): List of tensors with shape (token, channel).

    Returns:
        tuple:
            - torch.Tensor: Padded tensor with shape (batch, max_token, channel).
            - torch.Tensor: Mask tensor with shape (batch, max_token).
    """
    
    device = tensor_list[0].device
    # Pad and stack the data tensors
    padded_tensors = pad_sequence(tensor_list, batch_first=True).to(device)
    
    
    # Generate masks
    masks = [torch.ones(tensor.size(0), dtype=torch.bool, device=device) for tensor in tensor_list]
    
    padded_masks = pad_sequence(masks, batch_first=True, padding_value=0).to(device)
    
    
    
    return padded_tensors, padded_masks


def get_optimizer_and_scheduler(config, model, train_size):
    
    scheduler = None
    
    if config.optimizer.type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)
    elif config.optimizer.type == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)
    
    
    
    else:
        raise Exception("no optimizer")
    
    if config.scheduler.use_scheduler:
        
        if config.scheduler.type == 'CyclicLR_v1':
            scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=config.optimizer.lr, max_lr= config.optimizer.lr* 10,
                                                    cycle_momentum=False,
                                                    step_size_up= train_size // config.batch_size)
        
        elif config.scheduler.type == 'CyclicLR_v2':
            scheduler = optim.lr_scheduler.CyclicLR(
                optimizer, 
                base_lr=config.optimizer.lr, 
                max_lr=config.optimizer.lr * 10,
                step_size_up=int((train_size // config.batch_size) * 0.5), # 에포크의 x마다 맥스에 도달함
                cycle_momentum=False,
                mode='triangular2'
                )
        else:
            raise Exception("No scheduler version!")
    
    
    return optimizer, scheduler





'''
code reference: https://github.com/hkmztrk/DeepDTA

r_squared_error, get_k, squared_error_zero, get_rm2
'''

def r_squared_error(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))


def squared_error_zero(y_obs,y_pred):
    k = get_k(y_obs,y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k*y_pred)) * (y_obs - (k* y_pred)))
    down= sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig,ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2*r2)-(r02*r02))))