import argparse
import pickle as pkl
import xgboost as xgb

import torch
from torch.utils.data import DataLoader

from ..util import *
from .pred_util import *



args = argparse.ArgumentParser(description="Args for predictor training")

## for training process
args.add_argument("--data_in_path", type=str,   default="../gen_TL_data/data_gen_results/")
args.add_argument("--wl_type",      type=str,   default="conv_k3",  choices=["conv_k3", "conv_k1", "gemm"])

args.add_argument("--tree_num",     type=int,   default=400)
args.add_argument("--tree_depth",   type=int,   default=7,          choices=[2, 3, 4, 5, 6, 7])

args.add_argument("--model_type",   type=str,   default="onehot",   choices=["regressor", "onehot", "embedding"])
args.add_argument("--lr",           type=float, default=1e-4)

args = args.parse_args()





if __name__ == "__main__":
    # #####################################
    # ########## Part 1: Load data ########
    # #####################################

    data_all = pkl.load(open(args.data_in_path+args.wl_type+"_parsed.pkl", "rb"))
    X_train, X_test, y_train, y_test = data_all['X_train'], data_all['X_test'], data_all['y_train'], data_all['y_test']

    tree_num_bank = [100, 200, 400, 800, 1600, 3200]
    tree_depth_bank = [2, 3, 4, 5, 6, 7]

    mse_bank_all  = np.zeros((len(tree_num_bank), len(tree_depth_bank)))
    nmse_bank_all = np.zeros((len(tree_num_bank), len(tree_depth_bank)))

    save_path = "./pred_models/{:}_i48_cp/".format(args.wl_type)
    os.makedirs(save_path, exist_ok=True)


    # ##############################################
    # ########## Part 2: Train concate NNs #########
    # ##############################################
    # tree_num = 800
    # tree_depth = 7
    concate_list = [0, 1, 2, 3, 4]

    # step 1: get input data for nn
    leaf_train_all = []
    leaf_test_all  = []
    n_leaf_node = 0
    for xgb_id in concate_list:
        xgb_model_path = "../xgboost_search/pred_models/{:}_i48_n{:}_d{:}/".format(args.wl_type, args.tree_num, args.tree_depth)
        xgb_model_name = "xgb_{:}.model".format(xgb_id)
        model_xgb = xgb.Booster()
        model_xgb.load_model( xgb_model_path + xgb_model_name )

        leaf_train = model_xgb.predict(xgb.DMatrix(X_train), pred_leaf=True).astype(int)
        leaf_test  = model_xgb.predict(xgb.DMatrix(X_test ), pred_leaf=True).astype(int)
        leaf_train_all.append(leaf_train)
        leaf_test_all.append(leaf_test)
        n_leaf_node = max(n_leaf_node, np.max(leaf_train) + 1)


    # step 2: define NN params
    leaf_train = np.concatenate(leaf_train_all, axis=1)
    leaf_test  = np.concatenate(leaf_test_all,  axis=1)
    print(leaf_train.shape, leaf_test.shape)
    leaf_train = torch.tensor(leaf_train, dtype=torch.float)
    leaf_test  = torch.tensor(leaf_test,  dtype=torch.float)
    y_nn_train = torch.tensor(y_train, dtype=torch.float32)
    y_nn_test  = torch.tensor(y_test,  dtype=torch.float32)

    train_dataset = MyDataset(leaf_train, y_nn_train)
    val_dataset   = MyDataset(leaf_test, y_nn_test)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=128, shuffle=True)

    if args.model_type == "onehot":
        model_nn = OneHotRegressor(n_trees=leaf_train.shape[1], n_leafs=int(n_leaf_node), n_outputs=y_train.shape[1])
    else:
        raise(NotImplementedError)
    optimizer = torch.optim.Adam(model_nn.parameters(), lr=args.lr)


    # step 3: run train-eval
    print("=== Start running {:} concate and prune ===".format(model_nn.name))
    model_nn = model_nn.cuda()
    model_nn, mse_best, nmse_best = run_train_eval(model_nn, train_loader, val_loader, optimizer, L1_gamma=1e-7, epoch_num=400)
    save_name = "n{:}_d{:}_{:}.pth".format(leaf_train.shape[1], args.tree_depth, args.model_type)
    torch.save(model_nn.state_dict(), save_path+save_name)

    mse_bank = mse_best.numpy()
    nmse_bank = nmse_best.numpy()
    print("Tree#:{:>4d} depth:{:>1d}   MSE: ".format(leaf_train.shape[1], args.tree_depth), end="")
    for i in range(len(mse_bank)):
        print("{:.2e} ".format(mse_bank[i]), end="")
    print("  mean:{:.2e}".format(np.mean(mse_bank)))
    print("                    NMSE: ", end="")
    for i in range(len(nmse_bank)):
        print("{:.2e} ".format(nmse_bank[i]), end="")
    print("  mean:{:.2e}".format(np.mean(nmse_bank)))
    
    

    # step 4: run prune-finetune (if required), algorithm: L1 tree prune
    print("Starting prune-finetune iterations")
    global_keep_trees_idx = np.arange(leaf_train.shape[1])
    prune_target_bank = [3200, 1600, 800, 400, 200, 100]
    prune_counter = 0
    while prune_target_bank[prune_counter] > leaf_train.shape[1]:
            prune_counter += 1

    while leaf_train.shape[1] > prune_target_bank[-1]:
        prune_target = prune_target_bank[prune_counter]
        model_nn = model_nn.cpu()
        
        # step 4-1: compute the importance of tree
        with torch.no_grad():
            L1_norm = torch.norm(model_nn.mlp[0].weight, 1, dim=0).reshape(-1, model_nn.n_leafs)
        tree_l1_scores = L1_norm.sum(dim=1).detach().cpu().numpy()
        
        # step 4-2: prune the model based on L1 score (keep the tree region with max score)
        keep_trees_idx = np.argsort(-tree_l1_scores)[:prune_target]
        keep_idx = []
        for t in keep_trees_idx:
            keep_idx.extend(range(t * model_nn.n_leafs, (t + 1) * model_nn.n_leafs))
        keep_idx = np.array(keep_idx)
        global_keep_trees_idx = global_keep_trees_idx[keep_trees_idx]
        save_txt_name = "n{:}_d{:}_{:}_idx.txt".format(len(global_keep_trees_idx), args.tree_depth, args.model_type)
        np.savetxt(save_path+save_txt_name, global_keep_trees_idx, fmt='%d', delimiter=' ')

        with torch.no_grad():
            model_nn.mlp[0].weight = torch.nn.Parameter(model_nn.mlp[0].weight[:, keep_idx])
        
        leaf_train = leaf_train[:, keep_trees_idx ].clone().detach()
        leaf_test  = leaf_test[ :, keep_trees_idx ].clone().detach()
        train_dataset = MyDataset(leaf_train, y_nn_train)
        val_dataset   = MyDataset(leaf_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=128, shuffle=True)

        # step 4-3: finetune model
        optimizer = torch.optim.Adam(model_nn.parameters(), lr=args.lr)
        model_nn = model_nn.cuda()
        model_nn, mse_best, nmse_best = run_train_eval(model_nn, train_loader, val_loader, optimizer, L1_gamma=1e-7, epoch_num=100)

        # step 5: save pruned models
        save_name = "n{:}_d{:}_{:}.pth".format(leaf_train.shape[1], args.tree_depth, args.model_type)
        torch.save(model_nn.state_dict(), save_path+save_name)

        mse_bank = mse_best.numpy()
        nmse_bank = nmse_best.numpy()
        print("Tree#:{:>4d} depth:{:>1d}   MSE: ".format(leaf_train.shape[1], args.tree_depth), end="")
        for i in range(len(mse_bank)):
            print("{:.2e} ".format(mse_bank[i]), end="")
        print("  mean:{:.2e}".format(np.mean(mse_bank)))
        print("                    NMSE: ", end="")
        for i in range(len(nmse_bank)):
            print("{:.2e} ".format(nmse_bank[i]), end="")
        print("  mean:{:.2e}".format(np.mean(nmse_bank)))

        prune_counter += 1


    