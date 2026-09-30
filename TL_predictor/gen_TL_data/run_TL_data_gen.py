#### randomly generate the problem and architecture, and run timeloop ####
#### candidate layout: CPQ, PQC, QPC (narrow down to the three layout based on previous experiments) ####


import os
import argparse
import numpy as np
import pickle as pkl

from ..util import *

THIS_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


args = argparse.ArgumentParser(description="Problem Generator")

args.add_argument("--output_dir",   type=str, default="output")
args.add_argument("--working_path", type=str, default="")
args.add_argument("--save_dir",     type=str, defaut="data_gen_results")
args.add_argument("--clear", action="store_true", help="Clear output directories")

## for arch generator
args.add_argument("--arch_type", type=str, default="eyeriss", choices=["eyeriss"])  # "nvdla", "tpu"

args.add_argument("--Global_Buffer_N_Banks", type=int, default=32)

args.add_argument("--ifmap_spad_Depth",   type=int, default=4)
args.add_argument("--weights_spad_Depth", type=int, default=4)
args.add_argument("--ofmap_spad_Depth",   type=int, default=4)

args.add_argument("--wl_type", type=str, default="conv_k3", choices=["conv_k3", "conv_k1", "gemm"])
args.add_argument("--layout_type", type=int, default=0)

args = args.parse_args()





if __name__ == "__main__":

    ## archtecture setup

    PE_mesh_bank = [[16, 32], [32, 32], [32, 64], [64, 64], [64, 128]]  # PE_size = [512, 1024, 2048, 4096, 8192]
    Ubuf_width_bank = [64]
    Ubuf_size_bank = [256, 512, 1024, 2048, 4096]  # in KB

    args.arch_type = "eyeriss"
    args.Global_Buffer_N_Banks = 32

    args.ifmap_spad_Depth   = 4
    args.weights_spad_Depth = 4
    args.ofmap_spad_Depth   = 4

    layout_choice_bank = [0, 3]             # [0, 3, 5]
                                            # layout_ref_bank = [ "C, P, Q", "C, Q, P", "P, C, Q", "P, Q, C", "Q, C, P", "Q, P, C" ]
    B_choice_bank = 2 ** np.arange(7)       # Batch size = [1, 2, 4, 8, 16, 32, 64]

    if len(args.working_path) == 0:
        args.working_path = os.path.joint(THIS_SCRIPT_DIR, "default_"+args.wl_type)
    os.makedirs(args.working_path, exist_ok=True)
    save_path = os.path.joint(THIS_SCRIPT_DIR, args.save_dir)
    os.makddirs(save_path, exist_ok=True)



    # task list, dimension: [B, W_I, H_I, C_I, K_W, K_H, C_O]
    ## layout choice for timeloop

    if (args.wl_type == "conv_k3") or (args.wl_type == "conv_k1"):
        if args.wl_type == "conv_k3":
            K_W, K_H = 3, 3
        else:
            K_W, K_H = 1, 1

        W_choice_bank = np.zeros(224)   # sample from input=(224,224) image net
        W_size = 224
        for pool_iter in range(5):      # simualate pooled size
            for i in range(4):
                candidate_idx = (W_size+i) // (i+1)     # ceil of index
                W_choice_bank[candidate_idx-1] = 1
            W_size = (W_size+1) // 2
        W_choice_bank[2**np.arange(2, 7)-1] = 1
        W_choice_bank = (np.arange(224)+1)[W_choice_bank==1]
        H_choice_bank = W_choice_bank.copy()

        C_I_choice_bank = [3, 8, 16, 32, 64, 128, 256, 512]
        C_O_choice_bank = np.zeros(512)
        C_size = 512
        for pool_iter in range(5):      # simualate pooled size
            for i in range(32):
                candidate_idx = (C_size+i) // (i+1)     # ceil of index
                C_O_choice_bank[candidate_idx-1] = 1
            C_size = (C_size+1) // 2
        C_O_choice_bank = (np.arange(512)+1)[C_O_choice_bank==1]


    elif args.wl_type == "gemm":
        K_W, K_H = 1, 1

        W_choice_bank   = [1]
        H_choice_bank   = np.zeros(512)
        for i in range(32):     # assume max of distributed 32 cores
            candidate_idx = (512+i) // (i+1)    # ceil of index
            H_choice_bank[candidate_idx-1] = 1
        H_choice_bank = (np.arange(512)+1)[H_choice_bank==1]

        C_I_choice_bank = [8, 16, 32, 64, 128, 256, 512]    # max of transformer is 512
        C_O_choice_bank = np.zeros(2048)
        for i in range(32):
            candidate_idx = (2048+i) // (i+1)   # ceil of index
            C_O_choice_bank[candidate_idx-1] = 1
        C_O_choice_bank = (np.arange(2048)+1)[C_O_choice_bank==1]



    ## task generating

    comb_count = len(Ubuf_size_bank) * len(Ubuf_width_bank) * len(PE_mesh_bank) * \
            len(W_choice_bank) * len(H_choice_bank) * len(C_I_choice_bank) * len(C_O_choice_bank) * len(B_choice_bank)
    choice_bank = np.arange(comb_count)
    np.random.shuffle(choice_bank)

    task_list = []
    counter = 0
    while counter < comb_count and len(task_list) < 100000:
        value = choice_bank[counter]
        Ubuf_width = Ubuf_width_bank[value % len(Ubuf_width_bank)]
        Ubuf_size  = Ubuf_size_bank[ (value // len(Ubuf_width_bank)) % len(Ubuf_size_bank) ] * 1024
        PE_mesh = PE_mesh_bank[ (value // (len(Ubuf_width_bank) * len(Ubuf_size_bank))) % len(PE_mesh_bank) ]

        value = value // (len(Ubuf_width_bank) * len(Ubuf_size_bank) * len(PE_mesh_bank))
        
        # ## selection of task
        W   = W_choice_bank[    value %   len(W_choice_bank) ]
        H   = H_choice_bank[   (value //  len(W_choice_bank)) % len(H_choice_bank) ]
        C_I = C_I_choice_bank[ (value // (len(W_choice_bank)  * len(H_choice_bank))) % len(C_I_choice_bank) ]
        C_O = C_O_choice_bank[ (value // (len(W_choice_bank)  * len(H_choice_bank)   * len(C_I_choice_bank))) % len(C_O_choice_bank) ]
        B   = B_choice_bank[   (value // (len(W_choice_bank)  * len(H_choice_bank)   * len(C_I_choice_bank)   * len(C_O_choice_bank))) % len(B_choice_bank) ]

        W_I, H_I = W, H

        ifm_size = W_I * H_I * C_I * B
        fil_size = K_W * K_H * C_I * C_O
        ofm_size = W * H * C_O * B

        storage_size = ifm_size + fil_size + ofm_size

        if storage_size < Ubuf_size:
            task_list.append([ args_wl(B, W, H, C_I, K_W, K_H, C_O), 
                               args_arch(PE_mesh[0], PE_mesh[1], Ubuf_width, Ubuf_size) ])
        counter += 1
    
    print("total task: ", len(task_list))


    output_data = {}
    output_data["task_list"] = task_list



    ## run the task

    energy_bank, cycles_bank, E_comp_bank = [], [], []

    for i in range (len(task_list)):
        data_energy, data_cycles, data_E_comp = [], [], []
        for j in range(len(layout_choice_bank)):

            args.layout_type = layout_choice_bank[j]
            conv_problem(task_list[i][0], args.working_path)
            eyeriss_arch(task_list[i][1], args.working_path, args.layout_type)
            energy, cycles, E_comp = run_timeloop_temp(THIS_SCRIPT_DIR, args.working_path)

            data_energy.append(energy)
            data_cycles.append(cycles)
            data_E_comp.append(E_comp)
            
        energy_bank.append(data_energy)
        cycles_bank.append(data_cycles)
        E_comp_bank.append(data_E_comp)
        
        ## store results every 10 tasks
        if i % 10 == 0:
            output_data["energy"] = energy_bank
            output_data["cycles"] = cycles_bank
            output_data["E_comp"] = E_comp_bank
            save_file_name = os.path.join(save_path, args.wl_type+".pkl")
            pkl.dump(output_data, open( save_file_name, "wb"))


            

