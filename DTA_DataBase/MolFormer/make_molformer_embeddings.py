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

molformer_dict = {}
molformer_version = "ibm/MoLFormer-XL-both-10pct"
molformer = AutoModel.from_pretrained(molformer_version,  trust_remote_code=True)
molformer_tokenizer = AutoTokenizer.from_pretrained(molformer_version, trust_remote_code=True)



datasets = ['DAVIS', 'KIBA']

for dataset in datasets:
    
    dataset_path = os.path.join('..', 'datasets', dataset, 'drug_ids.pkl')
    
    with open(dataset_path, 'rb') as f:
        drug_dict = pickle.load(f)
    
    for k, v in drug_dict.items():
        print(f"processing {k}")
        smiles = v

        smiles_inputs = molformer_tokenizer(smiles, return_tensors='pt', padding= False)

        ids, masks = smiles_inputs['input_ids'], smiles_inputs['attention_mask']

        outputs = molformer(input_ids= ids, attention_mask= masks)

        smiles_embeddings = outputs['last_hidden_state'].detach()
        smiles_embeddings = smiles_embeddings.squeeze(0)
        smiles_embeddings.requires_grad = False

        #print(smiles_embeddings.shape)
        molformer_dict[str(k)] = smiles_embeddings.to('cpu')

    
    print(molformer_dict)
    
    torch.save(molformer_dict, f"{dataset}.pt")