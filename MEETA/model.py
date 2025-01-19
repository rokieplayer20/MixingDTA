import torch
import random
import torch.nn as nn
from torch.nn import init
from utils import *
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer
import math

class QKV(nn.Module):
    
    def __init__(self, config, input_dim):
        super().__init__()
        self.config = config
        
        assert (input_dim + config.n_embd) % 2 == 0, "(input_dim + config.n_embd) % 2 == 0"
        
        self.fc1 = nn.Linear(input_dim, (input_dim + config.n_embd) //2)
        self.activation = nn.SiLU()
        self.fc2 = nn.Linear((input_dim + config.n_embd) //2, config.n_embd)
        
        self.dropout = nn.Dropout(config.dropout_qkv)
        
        
        self.ln = nn.LayerNorm((input_dim + config.n_embd) //2)
        
    
    
    def forward(self, x):
        
        
        
        x = self.fc1(x)
        x = self.activation(x)
        
        
        x = self.ln(x)
        
        x = self.dropout(x)
        x = self.fc2(x)
        
        
        
        return x



class Regress(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.config = config
        
        
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, config.n_embd *4),
            nn.SiLU(),
            nn.GroupNorm(2 , config.n_embd *4),
            nn.Dropout(config.dropout_mlp),
            
            nn.Linear(config.n_embd *4, config.n_embd *2),
            nn.SiLU(),
            nn.GroupNorm(1 , config.n_embd *2),
            nn.Dropout(config.dropout_mlp),
            
            nn.Linear(config.n_embd *2, config.n_embd),
            
            
        )
        
        self.silu = nn.SiLU()
        
        self.out = nn.Linear(config.n_embd, 1)
        
        if config.binary_classification:
            self.sigmoid = nn.Sigmoid()
        
    
    
    def forward(self, x):
        
        binding_embedding = self.mlp(x)
        
        predicted_BA = self.out(self.silu(binding_embedding))
        
        if self.config.binary_classification:
            predicted_BA = self.sigmoid(predicted_BA)
        
        
        return predicted_BA, binding_embedding




class cross_AFT(nn.Module):
    """
    This code is modified from 'https://nn.labml.ai/transformers/aft/index.html'.
    """
    
    def __init__(self, config, drug_dim, protein_dim):
        super().__init__()
        self.config = config
        
        self.pos_bias = nn.Parameter(torch.zeros(config.max_seq_len, config.max_seq_len), requires_grad=config.cross_pos_bias_switch)
        self.Activation = nn.Sigmoid()
        
        self.query_d = nn.Linear(drug_dim, config.n_embd)
        self.key_p = nn.Linear(protein_dim, config.n_embd)
        self.value_p = nn.Linear(protein_dim, config.n_embd)
        
        
        self.query_p = nn.Linear(protein_dim, config.n_embd)
        self.key_d = nn.Linear(drug_dim, config.n_embd)
        self.value_d = nn.Linear(drug_dim, config.n_embd)
        
        self.proj_d = nn.Linear(config.n_embd, config.n_embd)
        self.proj_p = nn.Linear(config.n_embd, config.n_embd)
        
        
        
        
    
    
    def forward(self,
                smiles_embeddings, smiles_mask,
                protein_embeddings, protein_mask):
        
        B_d, T_d, C_d = smiles_embeddings.size()
        B_p, T_p, C_p = protein_embeddings.size()
        
        assert B_d == B_p
        
        
        smiles_embeddings = smiles_embeddings.permute(1,0,2)
        protein_embeddings = protein_embeddings.permute(1,0,2)
        
        if self.config.masking:
            
            
            smiles_mask_tmp = smiles_mask.unsqueeze(2)  # (B, T_d, 1)
            protein_mask_tmp = protein_mask.unsqueeze(1)  # (B, 1, T_p)
            
            
            
            combined_mask = smiles_mask_tmp * protein_mask_tmp  # (B, T_d, T_p)
            
            
            
            combined_mask = combined_mask.permute(1, 2, 0)  # (T_d, T_p, B)
        
        q_d = self.query_d(smiles_embeddings)
        k_p = self.key_p(protein_embeddings)
        v_p = self.value_p(protein_embeddings)
        
        pos_bias = self.pos_bias[:T_d, :T_p] # for cross-attention form
        pos_bias = pos_bias.unsqueeze(-1)
        
        if self.config.masking:
            pos_bias_d = pos_bias.masked_fill(combined_mask == 0, -1e9)
        else:
            pos_bias_d = pos_bias
        
        max_key_p = k_p.max(dim=0, keepdims=True)[0]
        max_pos_bias_d = pos_bias_d.max(dim=1,  keepdims=True)[0]
        
        exp_key_p = torch.exp(k_p - max_key_p)
        exp_pos_bias_d = torch.exp(pos_bias_d - max_pos_bias_d)
        
        num_d = torch.einsum('ijb,jbd->ibd', exp_pos_bias_d, exp_key_p * v_p)
        den_d = torch.einsum('ijb,jbd->ibd', exp_pos_bias_d, exp_key_p)
        
        y_d = self.Activation(q_d) * num_d / den_d
        y_d = self.proj_d(y_d)
        y_d = y_d.permute(1,0,2)
        
        ##########################################################
        q_p = self.query_p(protein_embeddings)
        k_d = self.key_d(smiles_embeddings)
        v_d = self.value_d(smiles_embeddings)
        
        pos_bias = self.pos_bias[:T_d, :T_p] # for cross-attention
        pos_bias = pos_bias.transpose(0,1)
        pos_bias = pos_bias.unsqueeze(-1)
        
        if self.config.masking:
            
            
            pos_bias_p = pos_bias.masked_fill(combined_mask.permute(1,0,2) == 0, -1e9)
        else:
            pos_bias_p = pos_bias
        
        max_key_d = k_d.max(dim=0, keepdims=True)[0]
        max_pos_bias_p = pos_bias_p.max(dim=1,  keepdims=True)[0]
        
        exp_key_d = torch.exp(k_d - max_key_d)
        exp_pos_bias_p = torch.exp(pos_bias_p - max_pos_bias_p)
        
        num_p = torch.einsum('ijb,jbd->ibd', exp_pos_bias_p, exp_key_d * v_d)
        den_p = torch.einsum('ijb,jbd->ibd', exp_pos_bias_p, exp_key_d)
        
        y_p = self.Activation(q_p) * num_p / den_p
        y_p = self.proj_p(y_p)
        y_p = y_p.permute(1,0,2) # (B, T, C)
        
        if self.config.masking:
            
            y_d = y_d.masked_fill(smiles_mask.unsqueeze(-1) ==0, -1e9)
            y_p = y_p.masked_fill(protein_mask.unsqueeze(-1) ==0, -1e9)
        
        y_d, _ = torch.max(y_d, dim=1)
        y_p, _ = torch.max(y_p, dim=1)
        
        
        
        return y_d, y_p





class DTA(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        if config.drug == 'MolFormer':
            self.drug_dim = 768
            self.drug_layer = 12
        else:
            raise Exception("No drug encoder")
        
        
        if config.protein == 'ESM3_open_small':
            self.protein_dim = 1536
            self.protein_layer = None
        elif config.protein == 'esm2_t6_8M_UR50D':
            self.protein_dim = 320
            self.protein_layer = 6
        elif config.protein == 'esm2_t12_35M_UR50D':
            self.protein_dim = 480
            self.protein_layer = 12
        else:
            raise Exception("No protein encoder")
        
        self.encoder_drug = QKV(config= config, input_dim=self.drug_dim)
        self.encoder_protein = QKV(config= config, input_dim=self.protein_dim)
        
        self.Block = cross_AFT(config= config,
                               drug_dim= config.n_embd, protein_dim= config.n_embd)
        
        self.regressor = Regress(config=config, input_dim= config.n_embd * 2)
        
    def forward(self, drug_embedding, protein_embedding,
               drug_mask, protein_mask):
        
        drug_embed = self.encoder_drug(drug_embedding)
        protein_embed = self.encoder_protein(protein_embedding)
        
        y_d, y_p = self.Block(drug_embed, drug_mask, protein_embed, protein_mask)
        
        
        y = torch.cat([y_d, y_p], dim=1)
        
        predicted_BA, binding_embedding = self.regressor(y)
        
        return predicted_BA, binding_embedding
    
    
    def forward_mixup(self,
                     drug_embedding1, protein_embedding1,
                     drug_mask1, protein_mask1,
                     drug_embedding2, protein_embedding2,
                     drug_mask2, protein_mask2, lambd):
        
        drug_embed1 = self.encoder_drug(drug_embedding1)
        protein_embed1 = self.encoder_protein(protein_embedding1)
        
        drug_embed2 = self.encoder_drug(drug_embedding2)
        protein_embed2 = self.encoder_protein(protein_embedding2)
        
        y_d1, y_p1 = self.Block(drug_embed1, drug_mask1, protein_embed1, protein_mask1)
            
        y_d2, y_p2 = self.Block(drug_embed2, drug_mask2, protein_embed2, protein_mask2)
        
        
        y1 = torch.cat([y_d1, y_p1], dim=1)
        y2 = torch.cat([y_d2, y_p2], dim=1)
        
        y = y1 * lambd + y2 * (1 - lambd)
        
        predicted_BA, binding_embedding = self.regressor(y)
        
        return predicted_BA, binding_embedding