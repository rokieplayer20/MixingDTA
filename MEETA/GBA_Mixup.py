import numpy as np
import copy
# import ipdb
import torch
import torch.nn as nn
import time
from torch.optim import Adam
from sklearn.neighbors import KernelDensity
from utils import *
import os
from tqdm import tqdm

def stats_values(targets):
    mean = np.mean(targets)
    min = np.min(targets)
    max = np.max(targets)
    std = np.std(targets)
    print(f'y stats: mean = {mean}, max = {max}, min = {min}, std = {std}')
    return mean, min, max, std


def GBA_Mixup(args, used_feature,mask_matrix = None,
              naive_GBA = True, reverse_mask = False, i_fold = -1):
    
    """
    The mask_matrix has a shape of [train_size, train_size].
    If there is no shared information between data points,
    the corresponding positions are masked with False.
    """
    
    if i_fold < 0:
        raise Exception("no fold")
    
    if args.dataset == 'KIBA' and reverse_mask:
        KIBA_sampling_prob_path = os.path.join(args.data_dir, 'sampling_prob',f'{args.dataset}_{i_fold}_{args.connection}_bw{args.kde_bandwidth}_reverse.npy')
    elif args.dataset == 'KIBA':
        KIBA_sampling_prob_path = os.path.join(args.data_dir, 'sampling_prob', f'{args.dataset}_{i_fold}_{args.connection}_bw{args.kde_bandwidth}.npy')
    
    
    if args.dataset == 'KIBA':
        
        
        if os.path.isfile(KIBA_sampling_prob_path):
            print(f"KIBA sampling prob {KIBA_sampling_prob_path}")
            print("Load KIBA sampling prob")
            
            
            mix_idx = np.load(KIBA_sampling_prob_path)
            
            return mix_idx
    
    
    if reverse_mask:
        mask_matrix = ~mask_matrix
    
    mix_idx = []
    y_list = used_feature
    
    is_np = isinstance(y_list,np.ndarray)
    if is_np:
        data_list = torch.tensor(y_list, dtype=torch.float32)
    else:
        data_list = y_list

    N = len(data_list)
    
    ######## use GBA_Mixup or Naive (all-pair case in DT Mixup pair sampling of paper) #######
    
    for i in tqdm(range(N)):
        
        # (1,) -> (1,1)
        # torch.Size([1]) -> torch.Size([1, 1])
        data_i = data_list[i]
        data_i = data_i.reshape(-1,data_i.shape[0]) # get 2D
        
        if args.show_process:
            if i % (N // 10) == 0:
                print('Mixup sample prepare {:.2f}%'.format(i * 100.0 / N ))
        
        ######### get kde sample rate ##########
        kd = KernelDensity(kernel=args.kde_type, bandwidth=args.kde_bandwidth).fit(data_i)  # should be 2D
        
        if naive_GBA:
            masked_data_list = data_list
            
            
        else:
            # (the number of datapoints, )
            mask = mask_matrix[i]
            mask_tns = torch.from_numpy(mask).reshape(-1,1)
            masked_data_list = data_list.masked_fill(~mask_tns, -1e9)
        
        each_rate = np.exp(kd.score_samples(masked_data_list))
        each_rate /= np.sum(each_rate)  # norm
        
        
        
        ####### visualization: observe relative rate distribution shot #######
        if args.show_process and i == 0:
            print(f'bw = {args.kde_bandwidth}')
            print(f'each_rate[:10] = {each_rate[:10]}')
            stats_values(each_rate)
            
        mix_idx.append(each_rate)
        
    mix_idx = np.array(mix_idx)

    self_rate = [mix_idx[i][i] for i in range(len(mix_idx))]

    if args.show_process:
        print(f'len(y_list) = {len(y_list)}, len(mix_idx) = {len(mix_idx)}, np.mean(self_rate) = {np.mean(self_rate)}, np.std(self_rate) = {np.std(self_rate)},  np.min(self_rate) = {np.min(self_rate)}, np.max(self_rate) = {np.max(self_rate)}')

    if args.save_sampling_prob and args.dataset == 'KIBA':
        
        print(f"save {KIBA_sampling_prob_path}")
        np.save(KIBA_sampling_prob_path, mix_idx)
        
    # ndarray (dataset, dataset)
    return mix_idx