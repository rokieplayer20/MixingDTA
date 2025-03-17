import os
import torch
import pickle
import numpy as np
import random
import gc
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL
import tqdm
from transformers import AutoModel, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"



def make_from_ESM3(prot, device):
    
    
    ret_dict = {}
    
    
    if device == 'cpu':
        client = ESM3.from_pretrained(ESM3_OPEN_SMALL, device= device)
    else:
        client = ESM3.from_pretrained(ESM3_OPEN_SMALL, device="cuda")
    
    
    for k, v in prot.items():
        print(f"processing {k}")
        
        protein = ESMProtein(
            sequence= v
        )
    
        protein_tensor = client.encode(protein)
    
        output = client.forward_and_sample(
            protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
            )
        
        
        
        ret = output.per_residue_embedding.detach()
        
        ret.requires_grad = False
        #ret = ret.squeeze()
        #output.per_residue_embedding.requires_grad_(False)
        ret.to('cpu')
        
        ret_dict[str(k)] = ret
        
        #print(ret.shape)
        
        del output
        
        if device == 'cpu':
            gc.collect()
        else:
            torch.cuda.empty_cache()
    
    return ret_dict


datasets = ['DAVIS', 'KIBA', 'BindingDB_Kd','PDBbind_Refined']


for dataset in datasets:
    
    dataset_path = os.path.join('..', 'datasets', dataset, 'protein_ids.pkl')
    
    with open(dataset_path, 'rb') as f:
        protein_dict = pickle.load(f)
    
    
    protein_embedding_dcit = make_from_ESM3(protein_dict, device='cpu')
    #print(protein_embedding_dcit)
    
    torch.save(protein_embedding_dcit, f"{dataset}.pt")