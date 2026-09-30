import os
import argparse
import numpy as np
import pickle as pkl

from sklearn.model_selection import train_test_split

import xgboost as xgb

import torch

from torch.utils.data import DataLoader

from ..util import *
from pred_util import *





args = argparse.ArgumentParser(description="Args for predictor training")

## for xgb feature extract
args.add_argument("--use_xgb_id",   type=int,   default=0)
args.add_argument("--tree_num",     type=int,   default=400)
args.add_argument("--tree_depth",   type=int,   default=7,      choices=[2, 3, 4, 5, 6, 7])

## for nn predictor
args.add_argument("--lr",           type=float, default=1e-4)
args.add_argument("--hid_dim",      type=int,   default=128)
args.add_argument("--model_type",   type=str,   default="regressor",    choices=["regressor", "onehot", "embedding"])

## for training process
args.add_argument("--data_in_path", type=str,   default="../gen_TL_data/data_gen_results/")
args.add_argument("--wl_type",      type=str,   default="conv_k3",  choices=["conv_k3", "conv_k1", "gemm"])

args = args.parse_args()





if __name__ == "__main__":

    #####################################
    ######## Part 0: Parsing Data #######
    #####################################
    
    data_path = args.data_in_path + args.wl_type + ".pkl"
    print("Loading data from " + data_path)
    data_all = pkl.load(open(data_path, 'rb'))

    data_feature = []
    data_label   = []

    for i in range(len(data_all['energy'])):
        feature = feature_flatten(data_all['task_list'][i])
        label, T_idx = label_flatten(data_all['task_list'][i], data_all['E_comp'][i], data_all['cycles'][i])
        data_feature.append(feature)
        data_label.append(label)

    data_feature = np.array(data_feature, dtype=np.float32)
    data_label   = np.array(data_label,   dtype=np.float32)
    label_scaler = (np.max(data_label, axis=0).reshape(-1))
    data_label   = data_label / label_scaler

    X_train, X_test, y_train, y_test = train_test_split(data_feature, data_label, test_size=0.3, random_state=42)



    #####################################
    ########## Part 1: Train xbg ########
    #####################################

    # step 1: define the model
    save_path = "./pred_models/" + args.wl_type + "/"
    print("Checking if the path " + save_path + "exist ...")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path+"mlp/", exist_ok=True)

    # step 2: train xgb model
    y_xgb_train = y_train / (np.max(y_train, axis=0).reshape(-1))
    model_xgb = None  # start empty
    round_param = { 'objective':'reg:squarederror', 'max_depth': args.tree_depth,
                    'learning_rate':0.01, 'subsample':0.9, 'colsample_bytree':0.8, 'eta':0.1, 'n_jobs':-1}
    
    print("Starting training xgb model")
    dtrain = xgb.DMatrix(X_train, label=y_train[:,args.use_xgb_id])
    n_trees = args.tree_num
    model_xgb = xgb.train(round_param, dtrain, num_boost_round=n_trees, xgb_model=model_xgb)

    ## step 3: save model as booster for C++ use
    model_xgb.save_model(save_path + "xgb.model")



    # #####################################
    # ########## Part 2: Train NNs ########
    # #####################################
    
    # step 1: get input data for nn
    leaf_train = model_xgb.predict(xgb.DMatrix(X_train), pred_leaf=True).astype(int)
    leaf_test  = model_xgb.predict(xgb.DMatrix(X_test ), pred_leaf=True).astype(int)
    n_leaf_node = np.max(leaf_train) + 1

    # step 2: define NN params
    leaf_train = torch.tensor(leaf_train, dtype=torch.float)
    leaf_test  = torch.tensor(leaf_test, dtype=torch.float)
    y_nn_train = torch.tensor(y_train, dtype=torch.float32)
    y_nn_test  = torch.tensor(y_test,  dtype=torch.float32)

    train_dataset = MyDataset(leaf_train, y_nn_train)
    val_dataset   = MyDataset(leaf_test, y_nn_test)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=128, shuffle=True)

    if args.model_type == "regressor":
        model_nn = DirectRegressor(input_dim=args.tree_num, output_dim=y_train.shape[1])
    elif args.model_type == "onehot":
        model_nn = OneHotRegressor(n_trees=args.tree_num, n_leafs=int(n_leaf_node), n_outputs=y_train.shape[1], hid_dim=args.hid_dim)
    elif args.model_type == "embedding":
        model_nn = TreeEmbeddingRegressor(n_trees=args.tree_num, n_leafs=int(n_leaf_node), embed_dim=8, n_outputs=y_train.shape[1])
    else:
        raise(NotImplementedError)
    optimizer = torch.optim.Adam(model_nn.parameters(), lr=args.lr)


    # step 3: run train-eval
    print("Starting training nn model from scratch")
    model_nn = model_nn.cuda()
    model_nn, mse_best, nmse_best = run_train_eval(model_nn, train_loader, val_loader, optimizer, L1_gamma=1e-7, epoch_num=400)

    mse_bank = mse_best.numpy()
    nmse_bank = nmse_best.numpy()
    print("Tree#:{:>4d} depth:{:>1d}   MSE: ".format(args.tree_num, args.tree_depth), end="")
    for i in range(len(mse_bank)):
        print("{:.2e} ".format(mse_bank[i]), end="")
    print("  mean:{:.2e}".format(np.mean(mse_bank)))
    print("                    NMSE: ", end="")
    for i in range(len(nmse_bank)):
        print("{:.2e} ".format(nmse_bank[i]), end="")
    print("  mean:{:.2e}".format(np.mean(nmse_bank)))


    # step 4: save model
    save_name = "n{:}_d{:}_xgb{:}_{:}.pth".format(args.tree_num, args.tree_depth, args.use_xgb_id, args.model_type[0])
    torch.save(model_nn.state_dict(), save_path+save_name)
    label_scaler.astype(np.float32).tofile(save_path + "output_bias.log")
    meta_data = np.array([args.tree_num, args.hid_dim, y_train.shape[1], n_leaf_node]).astype(np.float32).tofile(save_path + "meta.log")
    export_model(model_nn, save_path)