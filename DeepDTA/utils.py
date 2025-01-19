import torch
import numpy as np
import random
import pickle
import argparse
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import OrderedDict

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