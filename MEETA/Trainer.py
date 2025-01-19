import torch
import random
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader
from model import *
from utils import *
import pandas as pd
from dataset import *
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lifelines.utils import concordance_index 
from tqdm import tqdm
import pickle
from collections import OrderedDict
from GBA_Mixup import *

from scipy.stats import pearsonr, spearmanr
import time


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
        
        
        if config.loss_f == 'MSE':
            self.loss_f = nn.MSELoss()
        elif config.loss_f == 'HuberLoss':
            self.loss_f = nn.HuberLoss()
        elif config.loss_f == 'MAE':
            self.loss_f = nn.L1Loss()

        self.dataset_path = os.path.join(self.config.data_dir, 'datasets', self.config.dataset)
        with open(os.path.join(self.dataset_path, 'drug_id_2_idx.pkl'), 'rb') as file:
            self.drug_id2idx = pickle.load(file)
        with open(os.path.join(self.dataset_path, 'drug_idx_2_id.pkl'), 'rb') as file:
            self.drug_idx2id = pickle.load(file)
        
        with open(os.path.join(self.dataset_path, 'protein_id_2_idx.pkl'), 'rb') as file:
            self.protein_id2idx = pickle.load(file)
        with open(os.path.join(self.dataset_path, 'protein_idx_2_id.pkl'), 'rb') as file:
            self.protein_idx2id = pickle.load(file)

        
        
        self.train_df = pd.DataFrame(columns=['epoch', 'train_loss'])
        
        self.valid_df = pd.DataFrame(columns=['epoch', 'valid_loss',
                                              'MSE','RMSE', 'MAE', 'R2','CI', 'Rm2', 'Pearson', 'Spearman', 'AUPR'])
        
        self.test_df = pd.DataFrame(columns=['fold', 'test_loss',
                                              'MSE','RMSE', 'MAE', 'R2','CI', 'Rm2', 'Pearson', 'Spearman', 'AUPR'])
    
    
    def train(self):
        
        config = self.config
        
        datatset_root = os.path.join('..', 'DTA_DataBase', 'datasets', config.dataset)
        
        if self.dataset_path == datatset_root:
            pass
        else:
            raise Exception("no dataset path matching")
        
        mask_matrix_path = os.path.join('..', 'DTA_DataBase', 'mask_matrix')
        
        set_seed(config.seed)
        self.batch_size = self.config.batch_size
        self.mix_alpha = self.config.mix_alpha
        
        
        if not os.path.exists(config.result_root_path):
            os.makedirs(config.result_root_path)
        
        save_path = os.path.join(config.result_root_path, config.dataset)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        
        dataset_result_path = os.path.join(self.config.result_root_path, self.config.dataset)
        molformer_path = os.path.join(self.config.data_dir,self.config.drug, self.config.dataset) + '.pt'
        esm_path = os.path.join(self.config.data_dir,self.config.protein, self.config.dataset) + '.pt'
        #print(esm_path)
        set_seed(self.config.seed)
        self.batch_size = self.config.batch_size
        self.mix_alpha = self.config.mix_alpha
        
        mask_matrix_path = os.path.join('..', 'DTA_DataBase', 'mask_matrix')
        
        molformer_dict = torch.load(molformer_path, map_location= self.device)
        esm_dict = torch.load(esm_path, map_location= self.device)
        
        self.molformer_idx_dict = OrderedDict()
        self.esm_idx_dict = OrderedDict()
        
        
        for k, v in molformer_dict.items():
            self.molformer_idx_dict[self.drug_id2idx[k]] = v.to(self.device)
        
        for k,v in esm_dict.items():
            self.esm_idx_dict[self.protein_id2idx[k]] = v.to(self.device)
        
        
        if not os.path.exists(dataset_result_path):
            os.makedirs(dataset_result_path)
        
        for i_fold in range(1, self.config.n_splits + 1):
            
            save_path_i = os.path.join(dataset_result_path, f"{i_fold}_fold")
            early_stopper = EarlyStopping(patience= self.config.patience)
            
            with open(os.path.join(datatset_root, 'train_'+str(i_fold)+'.pkl'), 'rb') as file:
                train_set = pickle.load(file)
            with open(os.path.join(datatset_root, 'valid_'+str(i_fold)+'.pkl'), 'rb') as file:
                valid_set = pickle.load(file)
            with open(os.path.join(datatset_root, 'test_'+str(i_fold)+'.pkl'), 'rb') as file:
                test_set = pickle.load(file)
            
            
            MEETA = DTA(config= self.config).to(self.device)
            
            self.train_df = self.train_df.iloc[0:0]
            self.valid_df = self.valid_df.iloc[0:0]
            
            
            drug_train, protein_train, y_train = load_dataset(config= self.config, dataset=train_set,
                                                              drug_id_2_idx=self.drug_id2idx,
                                                              protein_id_2_idx=self.protein_id2idx)
            
            drug_valid, protein_valid, y_valid = load_dataset(config= self.config, dataset=valid_set,
                                                              drug_id_2_idx=self.drug_id2idx,
                                                              protein_id_2_idx=self.protein_id2idx)

            drug_test, protein_test, y_test = load_dataset(config= self.config, dataset=test_set,
                                                              drug_id_2_idx=self.drug_id2idx,
                                                              protein_id_2_idx=self.protein_id2idx)
            
            
            if config.is_mixup:
                
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
            
            drug_train, protein_train, y_train = drug_train.to(self.device), protein_train.to(self.device), y_train.to(self.device)
            drug_valid, protein_valid, y_valid = drug_valid.to(self.device), protein_valid.to(self.device), y_valid.to(self.device)
            drug_test, protein_test, y_test = drug_test.to(self.device), protein_test.to(self.device), y_test.to(self.device)
            
            valid_index_set = Index_dataset(drug_valid, protein_valid, y_valid)
            valid_index_loader = DataLoader(valid_index_set, batch_size=self.config.batch_size, shuffle= False)
            
            test_index_set = Index_dataset(drug_test, protein_test, y_test)
            test_index_loader = DataLoader(test_index_set, batch_size=self.config.batch_size, shuffle= False)
            
            iteration = len(y_train) // self.batch_size
            
            
            optimizer, scheduler = get_optimizer_and_scheduler(self.config, MEETA, len(y_train))
            
            for epoch in range(1, self.config.num_epochs):
                
                # train
                MEETA.train()
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
                        
                        drug_input_tmp_lst = [self.molformer_idx_dict[idx] for idx in drug_indices]
                        protein_input_tmp_lst = [self.esm_idx_dict[idx] for idx in protein_indices]

                        batch_drug, mask_drug = pad_tensor_list(drug_input_tmp_lst)
                        batch_protein, mask_protein  = pad_tensor_list(protein_input_tmp_lst)
                        
                        y_input = y_input_tmp

                        predicted_BA, binding_embedding = MEETA(batch_drug, batch_protein,
                                                                     mask_drug, mask_protein)
                        
                        train_loss = self.loss_f(predicted_BA, y_input)
                        train_losses_in_epoch.append(train_loss.item())

                        # backward
                        optimizer.zero_grad()
                        train_loss.backward()
                        optimizer.step()
                        if scheduler != None: # backward (without scheduler)
                            scheduler.step()
                
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
                        
                        drug_input_lst1 = [self.molformer_idx_dict[idx] for idx in drug_indices1]
                        drug_input_lst2 = [self.molformer_idx_dict[idx] for idx in drug_indices2]
                        protein_input_lst1 = [self.esm_idx_dict[idx] for idx in protein_indices1]
                        protein_input_lst2 = [self.esm_idx_dict[idx] for idx in protein_indices2]
                        
                        drug_tmp = len(drug_input_lst1)
                        protein_tmp = len(protein_input_lst1)
                        batch_drug, mask_drug = pad_tensor_list(drug_input_lst1 + drug_input_lst2)
                        batch_drug1, batch_drug2 = batch_drug[:drug_tmp], batch_drug[drug_tmp:]
                        mask_drug1, mask_drug2 = mask_drug[:drug_tmp], mask_drug[drug_tmp:]
                        
                        
                        batch_protein, mask_protein = pad_tensor_list(protein_input_lst1 + protein_input_lst2)
                        batch_protein1, batch_protein2 = batch_protein[:protein_tmp],batch_protein[protein_tmp:]
                        mask_protein1, mask_protein2 = mask_protein[:protein_tmp],mask_protein[protein_tmp:]
                        
                        # mixup
                        mixup_Y = y_train_input1 * lambd + y_train_input2 * (1 - lambd)
                        
                        predicted_BA, binding_embedding = MEETA.forward_mixup(batch_drug1, batch_protein1,
                                                                                   mask_drug1, mask_protein1,
                                                                                   batch_drug2, batch_protein2,
                                                                                   mask_drug2, mask_protein2, lambd)
                        
                        train_loss = self.loss_f(predicted_BA, mixup_Y)
                        train_losses_in_epoch.append(train_loss.item())
                        
                        optimizer.zero_grad()
                        train_loss.backward()
                        optimizer.step()
                        if scheduler != None: # backward (without scheduler)
                            scheduler.step()
                            
                train_loss_a_epoch = np.average(train_losses_in_epoch)
                # valid
                valid_pbar = tqdm(enumerate(valid_index_loader), total= len(valid_index_loader))
                score_dict, _ = self.test(model= MEETA, pbar = valid_pbar)
                
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
                    torch.save(MEETA.state_dict(), save_path_i + '_valid_best_checkpoint.pth')
                
                
                self.valid_df.to_csv(save_path_i + "_valid_history.csv", index=False)
                self.train_df.to_csv(save_path_i + "_train_history.csv", index=False)
                if early_stopper.early_stop:
                    
                    break
                
            # test
            MEETA.load_state_dict(torch.load(save_path_i + '_valid_best_checkpoint.pth'))
            
            test_pbar = tqdm(enumerate(test_index_loader), total= len(test_index_loader))
            test_dict, _ = self.test(model= MEETA, pbar = test_pbar)
            self.test_df.loc[len(self.test_df)] = [
                int(i_fold), test_dict['loss'], test_dict['MSE'], test_dict['RMSE'], test_dict['MAE'], test_dict['R2'], test_dict['CI'], test_dict['Rm2'],
                test_dict['Pearson'], test_dict['Spearman'], test_dict['AUPR']
            ]
            msg = f"{i_fold}_fold; test_loss({self.config.loss_f}): {test_dict['loss']:.3f}; \n \
                    MSE: {test_dict['MSE']:.3f}; RMSE: {test_dict['RMSE']:.3f}; R2: {test_dict['R2']:.3f}; MAE: {test_dict['MAE']:.3f}; Rm2: {test_dict['Rm2']:.3f}; CI: {test_dict['CI']:.3f};"
            print(msg)
            
            self.test_df.to_csv(os.path.join(dataset_result_path, "result.csv"), index=False)
            print(self.config)
        
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
    
    
    def test(self, model ,pbar):
        
        model.eval()
        
        scores = {}
        binding_embeddings = None
        
        losses_in_epoch = []
        total_preds = torch.Tensor()
        total_labels = torch.Tensor()
        
        with torch.no_grad():
            for i, data_i in pbar:
                drug_i, protein_i, label_i = data_i
                
                drug_indices = drug_i.cpu().numpy().flatten()
                protein_indices = protein_i.cpu().numpy().flatten()
                
                drug_input_tmp_lst = [self.molformer_idx_dict[idx] for idx in drug_indices]
                protein_input_tmp_lst = [self.esm_idx_dict[idx] for idx in protein_indices]

                batch_drug, mask_drug = pad_tensor_list(drug_input_tmp_lst)
                batch_protein, mask_protein  = pad_tensor_list(protein_input_tmp_lst)
                
                y_input = label_i

                predicted_BA, binding_embedding = model(batch_drug, batch_protein, mask_drug, mask_protein)

                loss = self.loss_f(predicted_BA, y_input.view(-1, 1))
                losses_in_epoch.append(loss.item())
                total_preds = torch.cat((total_preds, predicted_BA.cpu()), 0)
                total_labels = torch.cat((total_labels, y_input.cpu()), 0)
            
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

        return scores, binding_embeddings
    
    def integration(self):
        config = self.config
        print(f"Train integration.")
        
        
        self.config.pretrained_path= self.config.pretrained_path
        print(self.config.pretrained_path)
        pretrained_path = [ os.path.join('.', x, config.dataset) for x in self.config.pretrained_path]
        print(pretrained_path)
        
        dataset_result_path = os.path.join(self.config.result_integration, self.config.dataset)
        molformer_path = os.path.join(self.config.data_dir,self.config.drug, self.config.dataset) + '.pt'
        esm_path = os.path.join(self.config.data_dir,self.config.protein, self.config.dataset) + '.pt'
        
        set_seed(self.config.seed)
        self.batch_size = self.config.batch_size
        
        molformer_dict = torch.load(molformer_path, map_location= self.device)
        esm_dict = torch.load(esm_path, map_location= self.device)
        
        self.molformer_idx_dict = OrderedDict()
        self.esm_idx_dict = OrderedDict()
        
        for k, v in molformer_dict.items():
            self.molformer_idx_dict[self.drug_id2idx[k]] = v.to(self.device)
        
        for k,v in esm_dict.items():
            self.esm_idx_dict[self.protein_id2idx[k]] = v.to(self.device)
        
        if not os.path.exists(dataset_result_path):
            os.makedirs(dataset_result_path)
        
        if config.Integration_mode == 'embd':
            input_size = config.n_embd
        elif config.Integration_mode == 'pred':
            input_size = 1 
        elif config.Integration_mode == 'total':
            input_size = (1+config.n_embd)
        else:
            raise Exception("no Integration mode")
        
        for i_fold in range(1, self.config.n_splits + 1):
            
            save_path_i = os.path.join(dataset_result_path, f"{i_fold}_fold")
            
            pretrained_path_i = [ os.path.join(x, f"{i_fold}_fold_valid_best_checkpoint.pth") for x in pretrained_path]
            
            early_stopper = EarlyStopping(patience= self.config.patience)
            
            with open(os.path.join(self.dataset_path, 'train_'+str(i_fold)+'.pkl'), 'rb') as file:
                train_set = pickle.load(file)
            with open(os.path.join(self.dataset_path, 'valid_'+str(i_fold)+'.pkl'), 'rb') as file:
                valid_set = pickle.load(file)
            with open(os.path.join(self.dataset_path, 'test_'+str(i_fold)+'.pkl'), 'rb') as file:
                test_set = pickle.load(file)
            
            regressor = Regress(config= self.config, input_dim= input_size).to(self.device)
            
            
            self.train_df = self.train_df.iloc[0:0]
            self.valid_df = self.valid_df.iloc[0:0]
            
            drug_train, protein_train, y_train = load_dataset(config= self.config, dataset=train_set,
                                                              drug_id_2_idx=self.drug_id2idx,
                                                              protein_id_2_idx=self.protein_id2idx)
            
            drug_valid, protein_valid, y_valid = load_dataset(config= self.config, dataset=valid_set,
                                                              drug_id_2_idx=self.drug_id2idx,
                                                              protein_id_2_idx=self.protein_id2idx)

            drug_test, protein_test, y_test = load_dataset(config= self.config, dataset=test_set,
                                                              drug_id_2_idx=self.drug_id2idx,
                                                              protein_id_2_idx=self.protein_id2idx)
            
            drug_train, protein_train, y_train = drug_train.to(self.device), protein_train.to(self.device), y_train.to(self.device)
            drug_valid, protein_valid, y_valid = drug_valid.to(self.device), protein_valid.to(self.device), y_valid.to(self.device)
            drug_test, protein_test, y_test = drug_test.to(self.device), protein_test.to(self.device), y_test.to(self.device)
            
            train_index_set = Index_dataset(drug_train, protein_train, y_train)
            train_index_loader = DataLoader(train_index_set, batch_size=self.config.batch_size, shuffle= True)
            
            valid_index_set = Index_dataset(drug_valid, protein_valid, y_valid)
            valid_index_loader = DataLoader(valid_index_set, batch_size=self.config.batch_size, shuffle= False)
            
            test_index_set = Index_dataset(drug_test, protein_test, y_test)
            test_index_loader = DataLoader(test_index_set, batch_size=self.config.batch_size, shuffle= False)
            
            
            
            optimizer, scheduler = get_optimizer_and_scheduler(self.config, regressor, len(y_train))
            
            ### Load Trained Models ###
            Trained_models = []
            for pre_i in pretrained_path_i:
                tmp_model = DTA(config= self.config).to(self.device)
                tmp_model.load_state_dict(torch.load(pre_i, map_location=self.device))
                tmp_model.eval()
                Trained_models.append(tmp_model)
            
            for trained_model in Trained_models:
                for param in trained_model.parameters():
                    param.requires_grad = False
            
            
            for epoch in range(1, self.config.num_epochs):
                regressor.train()
                
                train_pbar = tqdm(enumerate(train_index_loader), total= len(train_index_loader))
                train_losses_in_epoch = []
                
                
                for train_i, train_data_i in train_pbar:
                    
                    train_tmp = None
                    
                    train_drug_i, train_protein_i, train_label_i = train_data_i
                    
                    train_drug_i = train_drug_i.cpu().numpy().flatten()
                    train_protein_i = train_protein_i.cpu().numpy().flatten()
                    
                    drug_input_tmp_lst = [self.molformer_idx_dict[idx] for idx in train_drug_i]
                    protein_input_tmp_lst = [self.esm_idx_dict[idx] for idx in train_protein_i]
                    
                    batch_drug, mask_drug = pad_tensor_list(drug_input_tmp_lst)
                    batch_protein, mask_protein  = pad_tensor_list(protein_input_tmp_lst)
                    
                    
                    y_input = train_label_i
                    
                    for trained_model in Trained_models:
                        
                        with torch.no_grad():
                            predicted_BA, binding_embedding = trained_model(batch_drug, batch_protein, mask_drug, mask_protein)
                        
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
                    
                    intergrated_BA, _ = regressor(train_tmp)
                    
                    train_loss = self.loss_f(intergrated_BA, y_input.view(-1,1))
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
                valid_pbar = tqdm(enumerate(valid_index_loader), total= len(valid_index_loader))
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
            
            
            test_pbar =tqdm(enumerate(test_index_loader), total= len(test_index_loader))
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
                
                test_tmp = None
                
                drug_i, protein_i, label_i = data_i
                
                drug_indices = drug_i.cpu().numpy().flatten()
                protein_indices = protein_i.cpu().numpy().flatten()
                
                drug_input_tmp_lst = [self.molformer_idx_dict[idx] for idx in drug_indices]
                protein_input_tmp_lst = [self.esm_idx_dict[idx] for idx in protein_indices]
                
                batch_drug, mask_drug = pad_tensor_list(drug_input_tmp_lst)
                batch_protein, mask_protein  = pad_tensor_list(protein_input_tmp_lst)
                
                
                y_input = label_i
                
                for trained_model in Trained_models:
                    predicted_BA, binding_embedding = trained_model(batch_drug, batch_protein, mask_drug, mask_protein)
                    
                    
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
                loss = self.loss_f(integrated_BA, y_input.view(-1,1))
                losses_in_epoch.append(loss.item())
                
                total_preds = torch.cat((total_preds, integrated_BA.cpu()), 0)
                total_labels = torch.cat((total_labels, y_input.cpu()), 0)
        
        
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