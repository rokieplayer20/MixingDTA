import torch
import numpy as np
import random
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_dataset(config, dataset, drug_id_2_idx, protein_id_2_idx):
    
    dataset_y = []
    dataset_drug_idxs = []
    dataset_protein_idxs = []
    
    for idx, data_i in enumerate(dataset):
        
        dataset_drug_idxs.append(drug_id_2_idx[str(data_i[0])])
        dataset_protein_idxs.append(protein_id_2_idx[str(data_i[2])])
        dataset_y.append(data_i[-1])
    
    Ys = torch.tensor(dataset_y, dtype=torch.float).view(-1,1)
    dataset_drug_idxs = torch.tensor(dataset_drug_idxs, dtype=torch.int32).view(-1,1)
    dataset_protein_idxs = torch.tensor(dataset_protein_idxs, dtype=torch.int32).view(-1,1)
    
    return dataset_drug_idxs, dataset_protein_idxs, Ys


class Index_dataset(Dataset):
    
    def __init__(self, drug, protein, label):
        
        self.drug = drug
        self.protein = protein
        self.label = label
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        
        return self.drug[idx], self.protein[idx], self.label[idx]


def find_connection(dataset_drug_idxs, dataset_protein_idxs, Ys, optional):
    
    """
    drug index, protein index
    train_list: [[0,1], [2,1], ....]
    len(train_list) == train_size
    
    return mask_matrix (train_size, train_size)
    
    """
    
    
    
    
    dataset = Index_dataset(dataset_drug_idxs, dataset_protein_idxs, Ys)
    
    loader = DataLoader(dataset= dataset, batch_size=1, shuffle=False)
    
    train_list = []
    
    for data_i in loader:
        
        drug, protein, _ = data_i

        train_list.append([drug, protein])
    
    
    
    train_size = len(train_list)
    
    mask_matrix = np.zeros((train_size,train_size), dtype= bool)
    
    
    for i in tqdm(range(train_size)):
        for j in range(train_size):
            
            if optional == 'all':
                if train_list[i][0] == train_list[j][0] or train_list[i][1] == train_list[j][1]:
                    mask_matrix[i, j] = True
            
            elif optional == 'drug':
                if train_list[i][0] == train_list[j][0]:
                    mask_matrix[i, j] = True
                    
            elif optional == 'protein':
                if train_list[i][1] == train_list[j][1]:
                    mask_matrix[i, j] = True
            else:
                
                raise Exception("no optional")
    
    
    return mask_matrix