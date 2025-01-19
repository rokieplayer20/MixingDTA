import os

none_case = {
    
    'result_root_path': "./results_none", # "path to store the results"
    
    'connection': 'nothing', 
    
    'naive_GBA': False,
    'is_mixup': False,
    'reverse_mask': False,
    
    
    
}
all_pair_case = {
    
    'result_root_path': "./results_all_pair", # "path to store the results"
    
    'connection': 'nothing', 
    
    'naive_GBA': True,
    'is_mixup': True,
    'reverse_mask': False,
    
    
    
}

drug_case = {
    
    'result_root_path': "./results_drug", # "path to store the results"
    
    'connection': 'drug', 
    
    'naive_GBA': False,
    'is_mixup': True,
    'reverse_mask': False,
    
    
    
}

protein_case = {
    
    'result_root_path': "./results_protein", # "path to store the results"
    
    'connection': 'protein', 
    
    'naive_GBA': False,
    'is_mixup': True,
    'reverse_mask': False,
    
    
    
}

drug_and_protein_case = {
    
    'result_root_path': "./results_drug_and_protein", # "path to store the results"
    
    'connection': 'all', 
    
    'naive_GBA': False,
    'is_mixup': True,
    'reverse_mask': False,
    
    
    
}

reversed_case = {
    
    'result_root_path': "./results_reversed", # "path to store the results"
    
    'connection': 'all', 
    
    'naive_GBA': False,
    'is_mixup': True,
    'reverse_mask': True,
    
    
    
}

cases = [none_case, all_pair_case, drug_case, protein_case, drug_and_protein_case, reversed_case]

configuration = {
    
    
    
    #'dataset': "DAVIS", # DAVIS, KIBA
    'data_dir': "../DTA_DataBase",
    
    'kde_type': 'gaussian', 
    'seed': 2024,
    'num_epochs': 500, 
    
    'n_splits': 5,
    'loss_f':"MSE", 
    'need_shuffle': True, 
    
    
    # model
    'binary_classification': False,  
    'smi_channels': 128,
    'seq_channels': 128,
    'num_filters':32,
    'window_smi': [4, 6, 8],
    'window_prot': [4, 8, 12],
    
    
    # train
    
    'batch_size': 64, # 64
    
    
    'kde_bandwidth': 21, #  default: 21
    'mix_alpha': 2.0,
    'patience': 30,
    'lr': 5e-5,
    
    
    
    # Integration (Stage 2)
    'weight_decay': 1e-4,
    # 'optimizer': {
    #         'type': 'AdamW', # Adam, AdamW
    #         'lr': 5e-5, # 5e-5
    #         'weight_decay': 1e-4,
    #     },
    
    # 'scheduler':{
    #         'use_scheduler': True,
    #         'type': 'CyclicLR_v2',
    #     },
    
    
    'Integration_mode': 'total', # embd, pred, total,
    #'patience_integration': 30,
    'result_integration': './result_integration',
    'pretrained_path': ['results_none', 'results_all_pair', 'results_drug', 'results_protein', 'results_drug_and_protein', 'results_reversed'],
    
    'n_mode': 6, # the number of pretrained_path
    
    'n_embd':512,
    
    'show_process': 1, # 'show rmse and r^2 in the process'
    
    'save_sampling_prob': False
    
    
}