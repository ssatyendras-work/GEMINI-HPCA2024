import os
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..util import *


##########################################
#### function for xgb feature extract ####
##########################################

def feature_flatten(data):
    def find_util(data_in):
        return [data_in/(((data_in+2**i-1)//(2**i))*(2**i)) for i in range(7)]
    
    wl = [ data[0].B, data[0].W, data[0].H, data[0].C_I, data[0].C_O ]
    arch = [ data[1].meshX, data[1].meshY, data[1].Ubuf_size ]
    
    ifmsize = data[0].B * data[0].W * data[0].H * data[0].C_I
    ofmsize = data[0].B * data[0].W * data[0].H * data[0].C_O
    wghsize = data[0].C_O * data[0].C_I * data[0].K_W * data[0].K_H
    
    ## fractions on meshX, applied factors (X): B, H, C_O  (Y): K_H, C_I, C_O
    assist = [ ifmsize, ofmsize, wghsize, float(np.log2(data[1].meshX)), float(np.log2(data[1].meshY)) ] + \
               find_util(data[0].B) + find_util(data[0].H) + find_util(data[0].C_O) + \
               find_util(data[0].K_H) + find_util(data[0].C_I)

    return wl + arch + assist


def label_flatten(spec, energy, cycle, target_layouts=-1):  # focus on the "per compute cost"
    # E_mac, E_weights, E_ifmap, E_psum, E_glb
    if target_layouts == -1:
        target_layouts = len(energy)

    T_total = (spec[0].B * spec[0].W * spec[0].H * spec[0].C_I * spec[0].C_O * spec[0].K_W * spec[0].K_H) / (spec[1].meshX*spec[1].meshY)
    label = []
    T_idx = []
    for layout_idx in range(target_layouts):
        label = label + [T_total/cycle[layout_idx]] + energy[layout_idx][1:] # removing E_mac since it is constant
        T_idx.append(layout_idx * len(energy[layout_idx]))
    return label, T_idx



#########################################
#### function for nn-based predictor ####
#########################################

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    

class DirectRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim=128):
        super().__init__()
        self.name = "regressor"
        self.hid_dim = hid_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.hid_dim),
            nn.BatchNorm1d(self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, output_dim)
        )
    def forward(self, x):
        return self.mlp(x)
    

class OneHotRegressor(nn.Module):
    def __init__(self, n_trees, n_leafs, n_outputs, hid_dim=128):
        super().__init__()
        self.name = "onehot"
        self.hid_dim = hid_dim
        self.n_leafs = n_leafs
        self.mlp = nn.Sequential(
            nn.Linear(n_trees * n_leafs, self.hid_dim),
            nn.BatchNorm1d(self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, n_outputs)
        )
    def forward(self, leaf_indices):
        leaf_indices = leaf_indices.to(torch.long)
        embeds = [F.one_hot(leaf_indices[:, i], num_classes=self.n_leafs) for i in range(leaf_indices.shape[1])]
        x = torch.cat(embeds, dim=1).to(torch.float)  # shape: (batch, n_trees * n_leafs)
        return self.mlp(x)


class TreeEmbeddingRegressor(nn.Module):
    def __init__(self, n_trees, n_leafs, embed_dim, n_outputs, hid_dim=128):
        super().__init__()
        self.name = "embedding"
        self.embed_dim = embed_dim
        self.hid_dim = hid_dim
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_leafs, embed_dim) for _ in range(n_trees)
        ])
        self.mlp = nn.Sequential(
            nn.Linear(n_trees * embed_dim, self.hid_dim),
            nn.BatchNorm1d(self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, n_outputs)
        )
    def forward(self, leaf_indices):
        leaf_indices = leaf_indices.to(torch.int)
        embeds = [self.embeddings[i](leaf_indices[:, i]) for i in range(len(self.embeddings))]
        x = torch.cat(embeds, dim=1)  # shape: (batch, n_trees * embed_dim)
        return self.mlp(x)



def run_train_eval(model, train_loader, val_loader, optimizer, L1_gamma=0, epoch_num=400):

    # loss_fn = nn.MSELoss()
    model_best = copy.deepcopy(model)
    mse_best = torch.tensor([float('inf')])
    nmse_best= torch.tensor([float('inf')])

    for epoch in range(epoch_num):

        # train
        model.train()
        for data, label in train_loader:
            data = data.cuda()
            label = label.cuda()

            pred = model(data)
            if model.name == "regressor":
                group_dim = 1
            elif model.name == "onehot":
                group_dim = model.n_leafs
            elif model.name == "embedding":
                group_dim = model.embed_dim
            L1_norm = torch.norm(model.mlp[0].weight, 1, dim=0).reshape(-1, group_dim)
            L1_norm = torch.mean(L1_norm.sum(dim=1))

            # nomalize the loss
            mse_loss = torch.mean( (pred-label)**2 )
            nmse_loss = torch.mean( ( (pred-label) / label )**2 )  # torch.mean( (pred-label)**2 )
            loss = (mse_loss + nmse_loss)/2 + L1_gamma*L1_norm
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # eval
        with torch.no_grad():
            model.eval()

            mse_epoch  = torch.zeros(label.shape[1])
            nmse_epoch = torch.zeros(label.shape[1])
            data_num  = 0
            for data, label in val_loader:
                data = data.cuda()
                val_pred = model(data)
                val_pred = val_pred.cpu()
                mse_epoch  += torch.sum( (val_pred - label)**2, axis=0 ).view(-1)
                nmse_epoch += torch.sum( ( (val_pred-label) / label )**2, axis=0 ).view(-1)
                data_num = data_num + val_pred.shape[0]
            mse_epoch  = mse_epoch  / data_num
            nmse_epoch = nmse_epoch / data_num
            L1_norm = torch.norm(model.mlp[0].weight, 1, dim=0)
            if model.name == "regressor":
                group_dim = 1
            elif model.name == "onehot":
                group_dim = model.n_leafs
            elif model.name == "embedding":
                group_dim = model.embed_dim
            L1_norm = L1_norm.reshape((-1,group_dim))
            L1_epoch = torch.mean( torch.sum(L1_norm, axis=1) )
            print("epoch: {:<3d}  MSE:{:.2e}  NMSE: {:.2e}  L1: {:.2e}     ".format(epoch+1, torch.mean(mse_epoch), torch.mean(nmse_epoch), L1_epoch*L1_gamma), end="\r")

            if (torch.sum(mse_epoch)+torch.sum(nmse_epoch)) < (torch.sum(mse_best)+torch.sum(nmse_best)):
                mse_best   = mse_epoch
                nmse_best  = nmse_epoch
                model_best = copy.deepcopy(model)
            
    # print("Overall  data#:{:}  MSE results: {:.2e}                                      ".format( data_num, torch.mean(mse_best)) )

    return model_best, mse_best, nmse_best




def save_tensor(tensor, file_path):
    """Save PyTorch tensor to binary float32 file"""
    np_array = tensor.detach().cpu().numpy().astype(np.float32)
    np_array.tofile(file_path)
    # print(f"Saved {file_path} shape={np_array.shape}")


def fold_bn_into_linear(fc, bn):
    """
    fc: nn.Linear layer
    bn: nn.BatchNorm1d layer
    Returns new folded weight & bias
    """
    W = fc.weight.detach().clone()  # shape: [out, in]
    b = fc.bias.detach().clone()    # shape: [out]

    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    mean = bn.running_mean.detach()
    var = bn.running_var.detach()
    eps = bn.eps

    std = torch.sqrt(var + eps)
    alpha = gamma / std
    beta_prime = beta - alpha * mean

    # Fold weights & bias
    W_folded = W * alpha[:, None]  # broadcast over in_dim
    b_folded = alpha * b + beta_prime

    return W_folded, b_folded


def export_model(model, export_dir, prefix=""):
    os.makedirs(export_dir, exist_ok=True)

    # Save first linear
    w1_fold, b1_fold = fold_bn_into_linear(model.mlp[0], model.mlp[1])
    
    save_tensor(w1_fold,    "{:}W1{:}.bin".format(export_dir, prefix))
    save_tensor(b1_fold,    "{:}b1{:}.bin".format(export_dir, prefix))

    # Save second linear
    fc2 = model.mlp[3]
    save_tensor(fc2.weight, "{:}W2{:}.bin".format(export_dir, prefix))
    save_tensor(fc2.bias,   "{:}b2{:}.bin".format(export_dir, prefix))

    print("- Model fully exported for C++ use.")