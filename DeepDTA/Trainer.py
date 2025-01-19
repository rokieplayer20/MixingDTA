import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lifelines.utils import concordance_index 
from tqdm import tqdm
import pickle
from model import *
from utils import *
import torch.utils.data as data
from torch.utils.data import DataLoader
#from torch.utils.data import DataLoader
import os
from random import shuffle
import sys
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from GBA_Mixup import *
from collections import OrderedDict
import random

class EarlyStopping:
    def __init__(self, patience, min_delta=0):
        
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0  
        else:
            self.counter += 1  
            if self.counter >= self.patience:
                self.early_stop = True  


class WorkStation:
    
    def __init__(self, config, device=None):
        
        self.config = config
        
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = device
        
        
        self.loss_f = nn.MSELoss()
        
        
        self.train_df = pd.DataFrame(columns=['epoch', 'train_loss'])
        
        
        
        self.valid_df = pd.DataFrame(columns=['epoch', 'valid_loss',
                                              'MSE','RMSE', 'MAE', 'R2','CI', 'Rm2', 'Pearson', 'Spearman', 'AUPR'])
        
        self.test_df = pd.DataFrame(columns=['fold', 'test_loss',
                                              'MSE','RMSE', 'MAE', 'R2','CI', 'Rm2', 'Pearson', 'Spearman', 'AUPR'])

        self.dataset_path = os.path.join(self.config.data_dir, 'datasets', self.config.dataset)
        with open(os.path.join(self.dataset_path, 'drug_id_2_idx.pkl'), 'rb') as file:
            self.drug_id2idx = pickle.load(file)
        with open(os.path.join(self.dataset_path, 'drug_idx_2_id.pkl'), 'rb') as file:
            self.drug_idx2id = pickle.load(file)
        with open(os.path.join(self.dataset_path, 'drug_ids.pkl'), 'rb') as file:
            self.drug_ids = pickle.load(file)
        
        
        
        with open(os.path.join(self.dataset_path, 'protein_id_2_idx.pkl'), 'rb') as file:
            self.protein_id2idx = pickle.load(file)
        with open(os.path.join(self.dataset_path, 'protein_idx_2_id.pkl'), 'rb') as file:
            self.protein_idx2id = pickle.load(file)
        with open(os.path.join(self.dataset_path, 'protein_ids.pkl'), 'rb') as file:
            self.protein_ids = pickle.load(file)
        
        
    
    def train(self):
        
        config = self.config
        
        datatset_root = os.path.join('..', 'DTA_DataBase', 'datasets', config.dataset)
        
        mask_matrix_path = os.path.join('..', 'DTA_DataBase', 'mask_matrix')
        
        set_seed(config.seed)
        self.batch_size = self.config.batch_size
        self.mix_alpha = self.config.mix_alpha
        
        
        if not os.path.exists(config.result_root_path):
            os.makedirs(config.result_root_path)
        
        save_path = os.path.join(config.result_root_path, config.dataset)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        for i_fold in range(1, config.n_splits+1):
            
            
            
            self.train_df = self.train_df.iloc[0:0]
            self.valid_df = self.valid_df.iloc[0:0]
            
            save_path_i = os.path.join(save_path, f"{i_fold}_fold")
            
            early_stopper = EarlyStopping(patience= self.config.patience)
            
            with open(os.path.join(datatset_root, 'train_'+str(i_fold)+'.pkl'), 'rb') as file:
                train_set = pickle.load(file)
            with open(os.path.join(datatset_root, 'valid_'+str(i_fold)+'.pkl'), 'rb') as file:
                valid_set = pickle.load(file)
            with open(os.path.join(datatset_root, 'test_'+str(i_fold)+'.pkl'), 'rb') as file:
                test_set = pickle.load(file)
            
            model = DeepDTA(config= self.config).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
            #train_dataset = DTA_Dataset(content= train_set, config= config)
            
            drug_train, protein_train, y_train = load_dataset(config= self.config, dataset=train_set,
                                                              drug_id_2_idx=self.drug_id2idx,
                                                              protein_id_2_idx=self.protein_id2idx)
            
            if self.config.is_mixup:
                
                if self.config.naive_GBA:
                    mixup_idx_sample_rate = GBA_Mixup(args = self.config, used_feature = y_train,
                                                      mask_matrix = None, naive_GBA = True, reverse_mask = self.config.reverse_mask, i_fold = i_fold)
                else:
                    print(f"find the connection with {self.config.connection}")
                    
                    if not os.path.exists(mask_matrix_path):
                        os.makedirs(mask_matrix_path)
                    
                    # If the mask_matrix already exists, ensure it is only accessed for lookup purposes.  
                    tmp = os.path.join(mask_matrix_path, f"{self.config.dataset}_{i_fold}_{self.config.connection}.npy")
                    
                    print(f"connection: {self.config.connection}")
                    
                    if os.path.isfile(tmp):
                        mask_matrix = np.load(tmp)
                        print("Load mask_matrix")
                    else:
                        mask_matrix = find_connection(drug_train, protein_train, y_train, optional= self.config.connection)
                        
                        np.save(tmp, mask_matrix)

                    
                    mixup_idx_sample_rate = GBA_Mixup(args = self.config, used_feature = y_train,
                                                      mask_matrix = mask_matrix, naive_GBA = False, reverse_mask = self.config.reverse_mask, i_fold = i_fold)
                
            else:
                mixup_idx_sample_rate = None
            
            
            iteration = len(y_train) // self.batch_size
            
            valid_dataset = DTA_Dataset(content= valid_set, config= config)
            test_dataset = DTA_Dataset(content= test_set, config= config)
            valid_loader = DataLoader(valid_dataset, batch_size= config.batch_size, shuffle = False)
            test_loader = DataLoader(test_dataset, batch_size= config.batch_size, shuffle = False)
            
            
            for epoch in range(1, self.config.num_epochs):
                # train
                model.train()
                print(f"train epoch {epoch}")
                shuffle_idx = np.random.permutation(np.arange(len(y_train)))
                
                if self.config.need_shuffle:
                    drug_train_input = drug_train[shuffle_idx]
                    protein_train_input = protein_train[shuffle_idx]
                    y_train_input = y_train[shuffle_idx]
                
                else:
                    drug_train_input = drug_train
                    protein_train_input = protein_train
                    y_train_input = y_train
                
                train_losses_in_epoch = []

                if not self.config.is_mixup:
                    
                    # no mixup
                    
                    for idx in tqdm(range(iteration)):
                        
                        drug_input_tmp = drug_train_input[idx * self.batch_size:(idx + 1) * self.batch_size]
                        protein_input_tmp = protein_train_input[idx * self.batch_size:(idx + 1) * self.batch_size]
                        y_input_tmp = y_train_input[idx * self.batch_size:(idx + 1) * self.batch_size]
                        
                        drug_indices = drug_input_tmp.cpu().numpy().flatten()
                        protein_indices = protein_input_tmp.cpu().numpy().flatten()
                        
                        # idx:smi
                        drug_input_tmp_lst = [self.drug_ids[self.drug_idx2id[idx]] for idx in drug_indices]
                        protein_input_tmp_lst = [self.protein_ids[self.protein_idx2id[idx]] for idx in protein_indices]
                        
                        drug_ = torch.tensor(label_smiles_batch(drug_input_tmp_lst, config.smilen, CHARISOSMISET) ,dtype=torch.long).to(self.device)
                        protein_ = torch.tensor(label_sequence_batch(protein_input_tmp_lst, config.seqlen, CHARPROTSET) ,dtype=torch.long).to(self.device)
                        
                        y_input = y_input_tmp.to(self.device)
                        
                        predicted_BA, _ = model(drug_, protein_)
                        
                        train_loss = self.loss_f(predicted_BA, y_input)
                        train_losses_in_epoch.append(train_loss.item())
                        
                        # backward
                        optimizer.zero_grad()
                        train_loss.backward()
                        optimizer.step()
                else:
                    
                    # mixup
                    
                    for idx in tqdm(range(iteration)):
                        lambd = np.random.beta(self.mix_alpha, self.mix_alpha)
                        
                        if self.config.need_shuffle:
                            idx_1 = shuffle_idx[idx * self.batch_size:(idx + 1) * self.batch_size]
                        else:
                            idx_1 = np.arange(len(y_train))[idx * self.batch_size:(idx + 1) * self.batch_size]
                        
                        
                        idx_2 = np.array(
                            [np.random.choice(np.arange(y_train.shape[0]), p = mixup_idx_sample_rate[sel_idx]) for sel_idx in idx_1]
                            )
                        
                        
                        drug_train_input1, drug_train_input2 = drug_train[idx_1], drug_train[idx_2]
                        protein_train_input1, protein_train_input2 = protein_train[idx_1], protein_train[idx_2]
                        y_train_input1, y_train_input2 = y_train[idx_1], y_train[idx_2]
                        
                        drug_indices1 = drug_train_input1.cpu().numpy().flatten()
                        drug_indices2 = drug_train_input2.cpu().numpy().flatten()
                        protein_indices1 = protein_train_input1.cpu().numpy().flatten()
                        protein_indices2 = protein_train_input2.cpu().numpy().flatten()
                        
                        
                        drug_input_tmp_lst1 = [self.drug_ids[self.drug_idx2id[idx]] for idx in drug_indices1]
                        drug_input_tmp_lst2 = [self.drug_ids[self.drug_idx2id[idx]] for idx in drug_indices2]
                        
                        drug_1 = torch.tensor(label_smiles_batch(drug_input_tmp_lst1, config.smilen, CHARISOSMISET) ,dtype=torch.long).to(self.device)
                        drug_2 = torch.tensor(label_smiles_batch(drug_input_tmp_lst2, config.smilen, CHARISOSMISET) ,dtype=torch.long).to(self.device)
                        
                        protein_input_tmp_lst1 = [self.protein_ids[self.protein_idx2id[idx]] for idx in protein_indices1]
                        protein_input_tmp_lst2 = [self.protein_ids[self.protein_idx2id[idx]] for idx in protein_indices2]
                        
                        protein_1 = torch.tensor(label_sequence_batch(protein_input_tmp_lst1, config.seqlen, CHARPROTSET) ,dtype=torch.long).to(self.device)
                        protein_2 = torch.tensor(label_sequence_batch(protein_input_tmp_lst2, config.seqlen, CHARPROTSET) ,dtype=torch.long).to(self.device)
                        
                        y_train_input1, y_train_input2 = y_train_input1.to(self.device), y_train_input2.to(self.device)
                        
                        
                        predicted_BA, _ = model.forward_mixup(drug_1, protein_1, drug_2, protein_2, lambd)
                        
                        mixup_Y = y_train_input1 * lambd + y_train_input2 * (1 - lambd)
                        
                        
                        train_loss = self.loss_f(predicted_BA, mixup_Y)
                        
                        optimizer.zero_grad()
                        train_loss.backward()
                        optimizer.step()
                        
                        train_losses_in_epoch.append(train_loss.item())
                        
                train_loss_a_epoch = np.average(train_losses_in_epoch)
                
                # valid
                
                valid_pbar = tqdm(enumerate(valid_loader), total = len(valid_loader))
                score_dict = self.test(model, valid_pbar)
                
                self.valid_df.loc[len(self.valid_df)] = [
                    int(epoch), score_dict['loss'], score_dict['MSE'], score_dict['RMSE'],
                    score_dict['MAE'], score_dict['R2'], score_dict['CI'], score_dict['Rm2'], score_dict['Pearson'], score_dict['Spearman'],
                    score_dict['AUPR']
                ]
                
                self.train_df.loc[len(self.train_df)] = [
                    int(epoch), train_loss_a_epoch
                ]
                
                msg = f"Epoch: {epoch}; train_loss({self.config.loss_f}): {train_loss_a_epoch:.3f} ; valid_loss({self.config.loss_f}): {score_dict['loss']:.3f}; \n \
                    MSE: {score_dict['MSE']:.3f}; RMSE: {score_dict['RMSE']:.3f}; MAE: {score_dict['MAE']:.3f}; R2: {score_dict['R2']:.3f}; Rm2: {score_dict['Rm2']:.3f}; CI: {score_dict['CI']:.3f};"
                
                print(msg)
                
                early_stopper(val_loss= score_dict['MSE'])
                
                if early_stopper.counter == 0:
                    torch.save(model.state_dict(), save_path_i + '_valid_best_checkpoint.pth')
                
                self.valid_df.to_csv(save_path_i + "_valid_history.csv", index=False)
                self.train_df.to_csv(save_path_i + "_train_history.csv", index=False)
                if early_stopper.early_stop:
                    break
                
            model.load_state_dict(torch.load(save_path_i + '_valid_best_checkpoint.pth'))
            
            print(f"test {i_fold}")
            
            
            test_dict = self.test(model, tqdm(enumerate(test_loader), total= len(test_loader)))
            self.test_df.loc[len(self.test_df)] = [
                int(i_fold), test_dict['loss'], test_dict['MSE'], test_dict['RMSE'], test_dict['MAE'], test_dict['R2'], test_dict['CI'], test_dict['Rm2'], test_dict['Pearson'], test_dict['Spearman'],
                test_dict['AUPR']
            ]
            
            msg = f"{i_fold}_fold; test_loss({self.config.loss_f}): {test_dict['loss']:.3f}; \n \
                    MSE: {test_dict['MSE']:.3f}; RMSE: {test_dict['RMSE']:.3f}; R2: {test_dict['R2']:.3f}; MAE: {test_dict['MAE']:.3f}; Rm2: {test_dict['Rm2']:.3f}; CI: {test_dict['CI']:.3f};"
            print(msg)
            
            self.test_df.to_csv(os.path.join(save_path, "result.csv"), index=False)
            
            del model
        
        test_loss_avg, test_loss_std = self.test_df['test_loss'].mean(), self.test_df['test_loss'].std()
        test_MSE_avg, test_MSE_std = self.test_df['MSE'].mean(), self.test_df['MSE'].std()
        test_RMSE_avg, test_RMSE_std = self.test_df['RMSE'].mean(), self.test_df['RMSE'].std()
        test_MAE_avg, test_MAE_std = self.test_df['MAE'].mean(), self.test_df['MAE'].std()
        test_R2_avg, test_R2_std = self.test_df['R2'].mean(), self.test_df['R2'].std()
        test_CI_avg, test_CI_std = self.test_df['CI'].mean(), self.test_df['CI'].std()
        test_Rm2_avg, test_Rm2_std = self.test_df['Rm2'].mean(), self.test_df['Rm2'].std()
        
        test_Pearson_avg, test_Pearson_std = self.test_df['Pearson'].mean(), self.test_df['Pearson'].std()
        test_Spearman_avg, test_Spearman_std = self.test_df['Spearman'].mean(), self.test_df['Spearman'].std()
        
        test_AUPR_avg, test_AUPR_std = self.test_df['AUPR'].mean(), self.test_df['AUPR'].std()
        
        self.test_df.loc[len(self.test_df)] = [
            "mean (std)",f"{test_loss_avg:.4f} ({test_loss_std:.4f})",
            f"{test_MSE_avg:.4f} ({test_MSE_std:.4f})", f"{test_RMSE_avg:.4f} ({test_RMSE_std:.4f})",
            f"{test_MAE_avg:.4f} ({test_MAE_std:.4f})", f"{test_R2_avg:.4f} ({test_R2_std:.4f})",
            f"{test_CI_avg:.4f} ({test_CI_std:.4f})", f"{test_Rm2_avg:.4f} ({test_Rm2_std:.4f})",
            f"{test_Pearson_avg:.4f} ({test_Pearson_std:.4f})", f"{test_Spearman_avg:.4f} ({test_Spearman_std:.4f})",
            f"{test_AUPR_avg:.4f} ({test_AUPR_std:.4f})"
        ]
        
        self.test_df.to_csv(os.path.join(save_path, "result.csv"), index=False)
    
    def test(self, model, pbar):
        
        
        model.eval()
        total_preds = torch.Tensor()
        total_labels = torch.Tensor()
        losses_in_epoch = []
        score_dict = {}
        
        with torch.no_grad():
            for batch_idx, data_i in pbar:
                
                first_col, second_col, smiles, protein_seq, BA = data_i
                smiles, protein_seq, BA = smiles.to(self.device), protein_seq.to(self.device), BA.to(self.device)
                predicted_BA, _ = model(smiles, protein_seq)
                
                loss = self.loss_f(predicted_BA, BA.view(-1,1)).to(dtype=torch.float)
                
                total_preds = torch.cat((total_preds, predicted_BA.cpu()), 0)
                total_labels = torch.cat((total_labels, BA.view(-1, 1).cpu()), 0)
                losses_in_epoch.append(loss.item())
                
        Y, P = total_labels.numpy().flatten(),total_preds.numpy().flatten()
        
        MSE = mean_squared_error(Y, P)
        RMSE = np.sqrt(MSE)
        MAE = mean_absolute_error(Y, P)
        R2 = r2_score(Y, P)
        CI = concordance_index(Y, P)
        Rm2 = get_rm2(Y, P)
        loss_a_epoch = np.average(losses_in_epoch)
        
        pearson_corr, pearson_p_value = pearsonr(Y, P)
        spearman_corr, spearman_p_value = spearmanr(Y, P)
        
        if self.config.dataset == 'DAVIS':
            AUPR = get_aupr_davis(Y, P)
        elif self.config.dataset == 'KIBA':
            AUPR = get_aupr_kiba(Y,P)
        else:
            raise Exception("No dataset")
        
        
        
        score_dict['MSE'] = MSE
        score_dict['RMSE'] = RMSE
        score_dict['MAE'] = MAE
        score_dict['R2'] = R2
        score_dict['CI'] = CI
        score_dict['Rm2'] = Rm2
        score_dict['loss'] = loss_a_epoch
        
        score_dict['Pearson'] = pearson_corr
        score_dict['Spearman'] = spearman_corr
        
        score_dict['AUPR'] = AUPR
        
        return score_dict
    
    
    def integration(self):
        config = self.config
        print(f"Train integration.")
        
        
        self.config.pretrained_path= self.config.pretrained_path
        pretrained_path = [ os.path.join('.', x, config.dataset) for x in self.config.pretrained_path]
        #print(pretrained_path)
        
        dataset_result_path = os.path.join(self.config.result_integration, self.config.dataset)
        
        set_seed(self.config.seed)
        self.batch_size = self.config.batch_size
        
        if not os.path.exists(dataset_result_path):
            os.makedirs(dataset_result_path)
        
        if config.Integration_mode == 'embd':
            input_size = config.n_embd
        elif config.Integration_mode == 'pred':
            input_size = 1
        elif config.Integration_mode == 'total':
            input_size = 1+config.n_embd
        else:
            raise Exception("no  Integration mode")
        
        for i_fold in range(1, config.n_splits+1):
            save_path_i = os.path.join(dataset_result_path, f"{i_fold}_fold")
            
            pretrained_path_i = [ os.path.join(x, f"{i_fold}_fold_valid_best_checkpoint.pth") for x in pretrained_path]
            
            early_stopper = EarlyStopping(patience= self.config.patience)
            
            self.train_df = self.train_df.iloc[0:0]
            self.valid_df = self.valid_df.iloc[0:0]
            
            with open(os.path.join(self.dataset_path, 'train_'+str(i_fold)+'.pkl'), 'rb') as file:
                train_set = pickle.load(file)
            with open(os.path.join(self.dataset_path, 'valid_'+str(i_fold)+'.pkl'), 'rb') as file:
                valid_set = pickle.load(file)
            with open(os.path.join(self.dataset_path, 'test_'+str(i_fold)+'.pkl'), 'rb') as file:
                test_set = pickle.load(file)
            
            train_dataset = DTA_Dataset(content= train_set, config= config)
            train_loader = DataLoader(train_dataset, batch_size= config.batch_size, shuffle = True)
            
            valid_dataset = DTA_Dataset(content= valid_set, config= config)
            test_dataset = DTA_Dataset(content= test_set, config= config)
            valid_loader = DataLoader(valid_dataset, batch_size= config.batch_size, shuffle = False)
            test_loader = DataLoader(test_dataset, batch_size= config.batch_size, shuffle = False)
            
            regressor = Meta_regressor(config= self.config, input_dim= input_size).to(self.device)
            
            optimizer = optim.AdamW(regressor.parameters(), lr=config.lr, weight_decay=config.weight_decay)
            train_size = len(train_dataset)
            
            scheduler = optim.lr_scheduler.CyclicLR(
                optimizer, 
                base_lr=config.lr, 
                max_lr=config.lr * 10,
                step_size_up=int((train_size // config.batch_size) * 0.5), # 에포크의 x마다 맥스에 도달함
                cycle_momentum=False,
                mode='triangular2'
                )
            
            ### Load Trained Models ###
            Trained_models = []
            for pre_i in pretrained_path_i:
                tmp_model = DeepDTA(config= self.config).to(self.device)
                tmp_model.load_state_dict(torch.load(pre_i, map_location=self.device))
                tmp_model.eval()
                Trained_models.append(tmp_model)
            
            for trained_model in Trained_models:
                for param in trained_model.parameters():
                    param.requires_grad = False
            
            for epoch in range(1, self.config.num_epochs):
                regressor.train()
                
                train_pbar = tqdm(enumerate(train_loader), total= len(train_loader))
                train_losses_in_epoch = []
                
                for train_i, train_data_i in train_pbar:
                    
                    #train_tmp = torch.Tensor().to(self.device)
                    train_tmp = None
                    
                    first_col, second_col, smiles, protein_seq, BA = train_data_i
                    smiles, protein_seq, BA = smiles.to(self.device), protein_seq.to(self.device), BA.to(self.device)
                    #print(Trained_models)
                    for trained_model in Trained_models:
                        
                        with torch.no_grad():
                            predicted_BA, binding_embedding = trained_model(smiles, protein_seq)
                        
                        
                        if config.Integration_mode == 'embd':
                            if train_tmp == None:
                                train_tmp = binding_embedding
                            else:
                                train_tmp += binding_embedding
                            
                        if config.Integration_mode == 'pred':
                            if train_tmp == None:
                                train_tmp = predicted_BA
                            else:
                                train_tmp += predicted_BA
                            
                        if config.Integration_mode == 'total':
                            if train_tmp == None:
                                train_tmp = torch.concat([binding_embedding, predicted_BA], dim =1)
                            else:
                                train_tmp += torch.concat([binding_embedding, predicted_BA], dim =1)
                    
                    
                    predicted_BA, _ = regressor(train_tmp)
                    #BA = BA.

                    train_loss = self.loss_f(predicted_BA, BA.view(-1,1)) #.to(dtype=torch.float)

                    train_losses_in_epoch.append(train_loss.item())
                    
                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()
                    if scheduler != None: # backward (without scheduler)
                        scheduler.step()
                    
                train_loss_a_epoch = np.average(train_losses_in_epoch)
                self.train_df.loc[len(self.train_df)] = [
                    int(epoch), train_loss_a_epoch
                    ]
                
                # valid
                valid_pbar = tqdm(enumerate(valid_loader), total= len(valid_loader))
                score_dict = self.test_regressor(valid_pbar, Trained_models, regressor)
                
                self.valid_df.loc[len(self.valid_df)] = [
                int(epoch), score_dict['loss'], score_dict['MSE'], score_dict['RMSE'],
                score_dict['MAE'], score_dict['R2'], score_dict['CI'], score_dict['Rm2'], score_dict['Pearson'], score_dict['Spearman'],
                score_dict['AUPR']
                ]
                
                msg = f"Epoch: {epoch}; train_loss({self.config.loss_f}): {train_loss_a_epoch:.3f} ; valid_loss({self.config.loss_f}): {score_dict['loss']:.3f}; \n \
                    MSE: {score_dict['MSE']:.3f}; RMSE: {score_dict['RMSE']:.3f}; MAE: {score_dict['MAE']:.3f}; R2: {score_dict['R2']:.3f}; Rm2: {score_dict['Rm2']:.3f}; CI: {score_dict['CI']:.3f};"
                
                print(msg)
                
                early_stopper(val_loss= score_dict['MSE'])
                
                if early_stopper.counter == 0:
                    torch.save(regressor.state_dict(), save_path_i + '_valid_best_checkpoint.pth')
                
                self.valid_df.to_csv(save_path_i + "_valid_history.csv", index=False)
                self.train_df.to_csv(save_path_i + "_train_history.csv", index=False)
                if early_stopper.early_stop:
                    break
            
            # test
            regressor.load_state_dict(torch.load(save_path_i + '_valid_best_checkpoint.pth', map_location=self.device))
            
            test_pbar =tqdm(enumerate(test_loader), total= len(test_loader))
            test_dict = self.test_regressor(test_pbar, Trained_models, regressor)
            self.test_df.loc[len(self.test_df)] = [
                int(i_fold), test_dict['loss'], test_dict['MSE'], test_dict['RMSE'], test_dict['MAE'], test_dict['R2'], test_dict['CI'], test_dict['Rm2'],
                test_dict['Pearson'], test_dict['Spearman'], test_dict['AUPR']
            ]
            msg = f"{i_fold}_fold; test_loss({self.config.loss_f}): {test_dict['loss']:.3f}; \n \
                    MSE: {test_dict['MSE']:.3f}; RMSE: {test_dict['RMSE']:.3f}; R2: {test_dict['R2']:.3f}; MAE: {test_dict['MAE']:.3f}; Rm2: {test_dict['Rm2']:.3f}; CI: {test_dict['CI']:.3f};"
            print(msg)
            
            self.test_df.to_csv(os.path.join(dataset_result_path, "result.csv"), index=False)
        
        test_loss_avg, test_loss_std = self.test_df['test_loss'].mean(), self.test_df['test_loss'].std()
        test_MSE_avg, test_MSE_std = self.test_df['MSE'].mean(), self.test_df['MSE'].std()
        test_RMSE_avg, test_RMSE_std = self.test_df['RMSE'].mean(), self.test_df['RMSE'].std()
        test_MAE_avg, test_MAE_std = self.test_df['MAE'].mean(), self.test_df['MAE'].std()
        test_R2_avg, test_R2_std = self.test_df['R2'].mean(), self.test_df['R2'].std()
        test_CI_avg, test_CI_std = self.test_df['CI'].mean(), self.test_df['CI'].std()
        test_Rm2_avg, test_Rm2_std = self.test_df['Rm2'].mean(), self.test_df['Rm2'].std()
        
        test_Pearson_avg, test_Pearson_std = self.test_df['Pearson'].mean(), self.test_df['Pearson'].std()
        test_Spearman_avg, test_Spearman_std = self.test_df['Spearman'].mean(), self.test_df['Spearman'].std()
        
        test_AUPR_avg, test_AUPR_std = self.test_df['AUPR'].mean(), self.test_df['AUPR'].std()
        
        test_msg = (
            f"test_loss_avg ({self.config.loss_f}): {test_loss_avg:.3f}({test_loss_std:.3f}); "
            f"test_MSE_avg: {test_MSE_avg:.3f}({test_MSE_std:.3f}); "
            f"test_RMSE_avg: {test_RMSE_avg:.3f}({test_RMSE_std:.3f}); "
            f"test_MAE_avg: {test_MAE_avg:.3f}({test_MAE_std:.3f}); "
            f"test_R2_avg: {test_R2_avg:.3f}({test_R2_std:.3f}); "
            f"test_CI_avg: {test_CI_avg:.3f}({test_CI_std:.3f}); "
            f"test_Rm2_avg: {test_Rm2_avg:.3f}({test_Rm2_std:.3f})"
            )
        
        print(test_msg)
        
        self.test_df.loc[len(self.test_df)] = ["mean (std)",f"{test_loss_avg:.4f} ({test_loss_std:.4f})",
                                               f"{test_MSE_avg:.4f} ({test_MSE_std:.4f})", f"{test_RMSE_avg:.4f} ({test_RMSE_std:.4f})",
                                               f"{test_MAE_avg:.4f} ({test_MAE_std:.4f})", f"{test_R2_avg:.4f} ({test_R2_std:.4f})",
                                               f"{test_CI_avg:.4f} ({test_CI_std:.4f})", f"{test_Rm2_avg:.4f} ({test_Rm2_std:.4f})",
                                               f"{test_Pearson_avg:.4f} ({test_Pearson_std:.4f})", f"{test_Spearman_avg:.4f} ({test_Spearman_std:.4f})",
                                               f"{test_AUPR_avg:.4f} ({test_AUPR_std:.4f})"
                                               ]
        
        
        self.test_df.to_csv(os.path.join(dataset_result_path, "result.csv"), index=False)
        
    
    def test_regressor(self, pbar, Trained_models, regressor):
        
        regressor.eval()
        losses_in_epoch = []
        scores = {}
        total_preds = torch.Tensor()
        total_labels = torch.Tensor()
        
        with torch.no_grad():
            for i, data_i in pbar:
                
                #test_tmp = torch.Tensor().to(self.device)
                
                test_tmp = None
                
                first_col, second_col, smiles, protein_seq, BA = data_i
                smiles, protein_seq = smiles.to(self.device), protein_seq.to(self.device)
                BA = BA.to(self.device)
                for trained_model in Trained_models:
                    
                    predicted_BA, binding_embedding = trained_model(smiles, protein_seq)
                    
                    if self.config.Integration_mode == 'embd':
                        if test_tmp == None:
                            test_tmp = binding_embedding
                        else:
                            test_tmp += binding_embedding

                    if self.config.Integration_mode == 'pred':
                        if test_tmp == None:
                            test_tmp = predicted_BA
                        else:
                            test_tmp += predicted_BA
                        
                        
                    
                    if self.config.Integration_mode == 'total':
                        if test_tmp == None:
                            test_tmp = torch.concat([binding_embedding, predicted_BA], dim =1)
                        else:
                            test_tmp += torch.concat([binding_embedding, predicted_BA], dim =1)
                
                
                integrated_BA, _ = regressor(test_tmp)
                loss = self.loss_f(integrated_BA, BA.view(-1,1))
                losses_in_epoch.append(loss.item())
                
                total_preds = torch.cat((total_preds, integrated_BA.cpu()), 0)
                total_labels = torch.cat((total_labels, BA.cpu()), 0)
        
        Y, P = total_labels.numpy().flatten(), total_preds.numpy().flatten()
        
        MSE = mean_squared_error(Y, P)
        RMSE = np.sqrt(MSE)
        MAE = mean_absolute_error(Y, P)
        R2 = r2_score(Y, P)
        CI = concordance_index(Y, P)
        Rm2 = get_rm2(Y, P)
        loss_a_epoch = np.average(losses_in_epoch)
        
        pearson_corr, pearson_p_value = pearsonr(Y, P)
        spearman_corr, spearman_p_value = spearmanr(Y, P)
        
        if self.config.dataset == 'DAVIS':
            AUPR = get_aupr_davis(Y, P)
        elif self.config.dataset == 'KIBA':
            AUPR = get_aupr_kiba(Y,P)
        else:
            raise Exception("No dataset")
        
        
        scores['MSE'] = MSE
        scores['RMSE'] = RMSE
        scores['MAE'] = MAE
        scores['R2'] = R2
        scores['CI'] = CI
        scores['Rm2'] = Rm2
        scores['loss'] = loss_a_epoch
        
        scores['Pearson'] = pearson_corr
        scores['Spearman'] = spearman_corr
        
        scores['AUPR'] = AUPR
        
        return scores
