import torch
import torch.nn as nn
from dataset import *




class Meta_regressor(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.config = config
        
        
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256 *4),
            nn.SiLU(),
            nn.GroupNorm(2 , 256 *4),
            nn.Dropout(0.15),
            
            nn.Linear(256 *4, 256 *2),
            nn.SiLU(),
            nn.GroupNorm(1 , 256 *2),
            nn.Dropout(0.15),
            
            nn.Linear(256 *2, 256),
            
            
        )
        
        self.silu = nn.SiLU()
        
        self.out = nn.Linear(256, 1)
        
        
        
    
    
    def forward(self, x):
        
        binding_embedding = self.mlp(x)
        
        predicted_BA = self.out(self.silu(binding_embedding))
        
        
        
        
        return predicted_BA, binding_embedding






class DeepDTA(nn.Module):
    def __init__(self, config):
    #def __init__(self, smi_len, smi_channels, seq_len, seq_channels, num_filters, filter_length):
        super().__init__()
        self.config = config
        
        self.drug_MAX_LENGH = self.config.smilen
        self.protein_MAX_LENGH = self.config.seqlen
        
        #print(CHARPROTLEN, CHARISOSMILEN)
        #self.smi_embd = nn.Embedding(CHARISOSMILEN +1, config.smi_channels, padding_idx = 0)
        #self.prot_embd = nn.Embedding(CHARPROTLEN +1, config.seq_channels, padding_idx = 0)
        self.smi_embd = nn.Embedding(100, config.smi_channels, padding_idx = 0)
        self.prot_embd = nn.Embedding(100, config.seq_channels, padding_idx = 0)
        self.smi_conv = nn.Sequential(
            nn.Conv1d(config.smi_channels, config.num_filters, config.window_smi[0], stride=1),
            nn.ReLU(),
            nn.Conv1d(config.num_filters, config.num_filters * 2, config.window_smi[1], stride=1),
            nn.ReLU(),
            nn.Conv1d(config.num_filters * 2, config.num_filters * 3, config.window_smi[2], stride=1),
            nn.ReLU(),
            
        )
        self.seq_conv = nn.Sequential(
            nn.Conv1d(config.seq_channels, config.num_filters, config.window_prot[0], stride=1),
            nn.ReLU(),
            nn.Conv1d(config.num_filters, config.num_filters * 2, config.window_prot[1], stride=1),
            nn.ReLU(),
            nn.Conv1d(config.num_filters * 2, config.num_filters * 3, config.window_prot[2], stride=1),
            nn.ReLU(),
            
        )
        
        self.Drug_max_pool = nn.MaxPool1d(self.drug_MAX_LENGH-config.window_smi[0] -config.window_smi[1]- config.window_smi[2]+3)
        self.Protein_max_pool = nn.MaxPool1d(self.protein_MAX_LENGH -config.window_prot[0] -config.window_prot[1] -config.window_prot[2] + 3)
        
        
        self.fc = nn.Sequential(
            nn.Linear(config.num_filters * 6, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 512),
            
        )
        
        self.relu = nn.ReLU()
        
        self.fin = nn.Linear(512, 1)
        
        

    def forward(self, x_smi, x_seq):
        x_smi = self.smi_embd(x_smi)
        x_seq = self.prot_embd(x_seq)
        
        x_smi, x_seq = x_smi.permute(0, 2, 1), x_seq.permute(0, 2, 1)
        
        x_smi = self.smi_conv(x_smi)
        x_seq = self.seq_conv(x_seq)
        
        #x_smi, x_seq = x_smi.permute(0, 2, 1), x_seq.permute(0, 2, 1)
        
        drugConv = self.Drug_max_pool(x_smi).squeeze(2)
        proteinConv = self.Protein_max_pool(x_seq).squeeze(2)
        #x_smi, _ = torch.max(x_smi, dim=1)
        #x_seq, _ = torch.max(x_seq, dim=1)
        
        joint = torch.cat((drugConv, proteinConv), dim=1)
        
        
        x = self.fc(joint)
        
        y = self.fin(self.relu(x))
        
        return y, x # y가 예측값이고 x를 mixing 할 것이다.
    
    
    def forward_mixup(self, x_smi1, x_seq1, x_smi2, x_seq2, lambd):
        
        
        x_smi1 = self.smi_embd(x_smi1)
        x_seq1 = self.prot_embd(x_seq1)
        
        x_smi2 = self.smi_embd(x_smi2)
        x_seq2 = self.prot_embd(x_seq2)
        
        x_smi1, x_seq1 = x_smi1.permute(0, 2, 1), x_seq1.permute(0, 2, 1)
        x_smi2, x_seq2 = x_smi2.permute(0, 2, 1), x_seq2.permute(0, 2, 1)
        
        x_smi1 = self.smi_conv(x_smi1)
        x_seq1 = self.seq_conv(x_seq1)
        
        x_smi2 = self.smi_conv(x_smi2)
        x_seq2 = self.seq_conv(x_seq2)
        
        drugConv1 = self.Drug_max_pool(x_smi1).squeeze(2)
        proteinConv1 = self.Protein_max_pool(x_seq1).squeeze(2)
        
        drugConv2 = self.Drug_max_pool(x_smi2).squeeze(2)
        proteinConv2 = self.Protein_max_pool(x_seq2).squeeze(2)
        
        joint1 = torch.cat((drugConv1, proteinConv1), dim=1)
        
        joint2 = torch.cat((drugConv2, proteinConv2), dim=1)
        
        
        joint = joint1 * lambd + joint2 * (1-lambd)
        
        x = self.fc(joint)
        
        y = self.fin(self.relu(x))
        
        return  y, x

