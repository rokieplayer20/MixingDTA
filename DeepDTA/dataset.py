import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import random
import pickle

CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
				"U": 19, "T": 20, "W": 21, 
				"V": 22, "Y": 23, "X": 24, 
				"Z": 25 }

CHARPROTLEN = 25

CHARCANSMISET = { "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6, 
			 ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12, 
			 "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18, 
			 "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
			 "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30, 
			 "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36, 
			 "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42, 
			 "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48, 
			 "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54, 
			 "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
			 "t": 61, "y": 62}

CHARCANSMILEN = 62

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2, 
				"1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6, 
				"9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43, 
				"D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13, 
				"O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51, 
				"V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56, 
				"b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60, 
				"l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64



def one_hot_smiles(line, MAX_SMI_LEN, smi_ch_ind):
	X = np.zeros((MAX_SMI_LEN, len(smi_ch_ind))) #+1

	for i, ch in enumerate(line[:MAX_SMI_LEN]):
		X[i, (smi_ch_ind[ch]-1)] = 1 

	return X #.tolist()

def one_hot_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
	X = np.zeros((MAX_SEQ_LEN, len(smi_ch_ind))) 
	for i, ch in enumerate(line[:MAX_SEQ_LEN]):
		X[i, (smi_ch_ind[ch])-1] = 1

	return X #.tolist()


def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
	X = np.zeros(MAX_SMI_LEN)
	for i, ch in enumerate(line[:MAX_SMI_LEN]): #	x, smi_ch_ind, y
		X[i] = smi_ch_ind[ch]

	return X #.tolist()

def label_smiles_batch(smiles_list, MAX_SMI_LEN, smi_ch_ind):
    
    batch_size = len(smiles_list)
    
    Xs = np.zeros((batch_size, MAX_SMI_LEN), dtype=np.int64)
    
    for idx,smi in enumerate(smiles_list):
        x = label_smiles(smi, MAX_SMI_LEN, smi_ch_ind)
        Xs[idx] = x
    
    
    return Xs



def label_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
	X = np.zeros(MAX_SEQ_LEN)

	for i, ch in enumerate(line[:MAX_SEQ_LEN]):
		X[i] = smi_ch_ind[ch]

	return X #.tolist()


def label_sequence_batch(protein_list, MAX_SEQ_LEN, smi_ch_ind):
    
    batch_size = len(protein_list)
    
    Xs = np.zeros((batch_size, MAX_SEQ_LEN), dtype=np.int64)
    
    
    for idx,seq in enumerate(protein_list):
        x = label_sequence(seq, MAX_SEQ_LEN, smi_ch_ind)
        Xs[idx] = x
    
    
    return Xs


class DTA_Dataset(Dataset):
    """
    minibatch마다 새로이 토크나이징 한다. 그러면 최대 길이에 맞추어서 패딩할
    필요가 없다.
    
    """
    def __init__(self, content: list, config=None):
        
        self.config = config
        
        self.content = content
        
        # ibm/MoLFormer-XL-both-10pct
        # <bos>:0, <eos>:1, <pad>:2
        
        
        
        
        # esm2 side
        # <cls>:0, <pad>:1, <eos>:2
        #self.esm2_tokenizer = AutoTokenizer.from_pretrained(self.config.esm2)
        
        
        ### TODO : ADD SMILES TYPE CHOICE HERE
        self.SEQLEN = config.seqlen
        self.SMILEN = config.smilen
        #self.NCLASSES = n_classes
        self.charseqset = CHARPROTSET
        self.charseqset_size = CHARPROTLEN

        self.charsmiset = CHARISOSMISET ###HERE CAN BE EDITED
        self.charsmiset_size = CHARISOSMILEN
        #self.PROBLEMSET = setting_no
        
    
    def __len__(self):
        
        return len(self.content)
    
    
    
    def __getitem__(self, idx):
        '''
        sampling 할 때, id들이 추적되도록 만들었다.
        ligand_id = self.content[idx][0]
        protein_id = self.content[idx][1] 
        smiles = self.content[idx][2] -> torch.Tensor
        protein_seq = self.content[idx][3] -> torch.Tensor
        BA = self.content[idx][4]
        
        
        # DTA_list.append([row['Drug_ID'], smi, row['Target_ID'], row['Target'], row['Y']])
        
        '''
        
        
        ligand_id, smiles, protein_id, protein_seq, BA = self.content[idx]
        
        
        smiles = label_smiles(smiles, self.SMILEN, self.charsmiset)
        protein_seq = label_sequence(protein_seq, self.SEQLEN, self.charseqset)
        
        smiles = torch.tensor(smiles ,dtype=torch.long)
        protein_seq = torch.tensor(protein_seq, dtype=torch.long)
        
        BA = torch.tensor(BA, dtype=torch.float)
        
        return ligand_id, protein_id, smiles, protein_seq, BA


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
    약물 index, 단백질 index
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
    
