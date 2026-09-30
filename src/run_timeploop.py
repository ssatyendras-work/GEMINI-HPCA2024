import os
import argparse
import numpy as np

THIS_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

import timeloopfe.v4 as tl


args = argparse.ArgumentParser(description="Simulation Generator")

args.add_argument("--output_dir", type=str, default="output")
args.add_argument("--clear", action="store_true", help="Clear output directories")

################
## args for problem type, including (GEMM and CONV)
args.add_argument("--problem_type", type=str, default="CONV", choices=["GEMM", "CONV"])

## nomenclature of args for representation in timeloop
## args:      Batch,  W_I, H_I, C_I, C_O, K_W, K_H,  S_W,     S_H,     D_W,       D_H
## timeloop:  N,      P,   Q,   C,   M,   R,   S,    Wstride, Hstride, Wdilation, Hdilation

## args for input size
args.add_argument("--Batch", type=int, default=1)
args.add_argument("--W_I", type=int, default=128)
args.add_argument("--H_I", type=int, default=128)
args.add_argument("--C_I", type=int, default=3)

## args for output size
args.add_argument("--C_O", type=int, default=16)

## args for CONV kernel size
args.add_argument("--K_W", type=int, default=3)
args.add_argument("--K_H", type=int, default=3)
args.add_argument("--S_W", type=int, default=1)  # stride
args.add_argument("--S_H", type=int, default=1)
args.add_argument("--D_W", type=int, default=1)  # dilation
args.add_argument("--D_H", type=int, default=1)


################
## args for architecture type, including (NVDLA, Eyeriss, TPU)
args.add_argument("--arch_type", type=str, default="eyeriss", choices=["eyeriss", "nvdla", "tpu"])

## args for global buffer, can be used for NVDLA, Eyeriss, and TPU.
args.add_argument("--Global_Buffer_Depth", type=int, default=16384)
args.add_argument("--Global_Buffer_Width", type=int, default=256)
args.add_argument("--Global_Buffer_N_Banks", type=int, default=32)
args.add_argument("--Global_Buffer_Read_Bandwidth", type=int, default=32)
args.add_argument("--Global_Buffer_Write_Bandwidth", type=int, default=32)

## args for PE array
args.add_argument("--meshX", type=int, default=16)
args.add_argument("--meshY", type=int, default=16)

# === args for Eyeriss
args.add_argument("--ifmap_spad_Depth", type=int, default=16)
args.add_argument("--ifmap_spad_Width", type=int, default=8)
args.add_argument("--weights_spad_Depth", type=int, default=256)
args.add_argument("--weights_spad_Width", type=int, default=8)
args.add_argument("--ofmap_spad_Depth", type=int, default=64)
args.add_argument("--ofmap_spad_Width", type=int, default=8)

args.add_argument("--ofmap_seq",   type=str, default="M")
args.add_argument("--weights_seq", type=str, default="CR")
args.add_argument("--ifmap_seq",   type=str, default="")
args.add_argument("--ofmap_base", type=int, default=0)
args.add_argument("--weights_base", type=int, default=0)
args.add_argument("--ifmap_base", type=int, default=0)



args = args.parse_args()




#########################################
#### function for problem generation ####
#########################################

def conv_problem(args):
    write_path = os.path.join(THIS_SCRIPT_DIR, "prob/prob.yaml")

    with open(write_path, "w") as f:
        f.write("problem:" + "\n")
        f.write("  version: 0.4" + "\n")
        
        f.write("  instance:" + "\n")
        f.write("    N: " + str(args.Batch) + "\n")
        f.write("    P: " + str(args.W_I) + "\n")
        f.write("    Q: " + str(args.H_I) + "\n")
        f.write("    C: " + str(args.C_I) + "\n")
        f.write("    M: " + str(args.C_O) + "\n")
        f.write("    R: " + str(args.K_W) + "\n")
        f.write("    S: " + str(args.K_H) + "\n")
        f.write("    Wstride: " + str(args.S_W) + "\n")
        f.write("    Hstride: " + str(args.S_H) + "\n")
        f.write("    Wdilation: " + str(args.D_W) + "\n")
        f.write("    Hdilation: " + str(args.D_H) + "\n")

        f.write("  shape:" + "\n")
        f.write("    coefficients:" + "\n")
        f.write("    - default: 1" + "\n")
        f.write("      name: Wstride" + "\n")
        f.write("    - default: 1" + "\n")
        f.write("      name: Hstride" + "\n")
        f.write("    - default: 1" + "\n")
        f.write("      name: Wdilation" + "\n")
        f.write("    - default: 1" + "\n")
        f.write("      name: Hdilation" + "\n")

        f.write("    data_spaces:" + "\n")

        f.write("    - name: Weights" + "\n")  # weights = C_O x C_I x K_W x K_H
        f.write("      projection:" + "\n")
        f.write("      - - - C" + "\n")
        f.write("      - - - M" + "\n")
        f.write("      - - - R" + "\n")
        f.write("      - - - S" + "\n")

        f.write("    - name: Inputs" + "\n")    # input = Batch x C_I x W_I x H_I, with convolution with kernel size K_W x K_H
        f.write("      projection:" + "\n")
        f.write("      - - - N" + "\n")
        f.write("      - - - C" + "\n")
        f.write("      - - - R" + "\n")
        f.write("          - Wdilation" + "\n")
        f.write("        - - P" + "\n")
        f.write("          - Wstride" + "\n")
        f.write("      - - - S" + "\n")
        f.write("          - Hdilation" + "\n")
        f.write("        - - Q" + "\n")
        f.write("          - Hstride" + "\n")

        f.write("    - name: Outputs" + "\n")   # output = Batch x C_O x W_O x H_O
        f.write("      projection:" + "\n")
        f.write("      - - - N" + "\n")
        f.write("      - - - M" + "\n")
        f.write("      - - - Q" + "\n")
        f.write("      - - - P" + "\n")
        f.write("      read_write: true" + "\n")

        f.write("    dimensions:" + "\n")
        f.write("    - C" + "\n")
        f.write("    - M" + "\n")
        f.write("    - N" + "\n")
        f.write("    - P" + "\n")
        f.write("    - Q" + "\n")
        f.write("    - R" + "\n")
        f.write("    - S" + "\n")
        f.write("    name: CNN_Layer" + "\n")




##############################################
#### function for architecture generation ####
##############################################

## generate EyeRiss architecture, constraints, and mapping
def eyeriss_arch(args):

    ## generate eyeriss architecture
    write_path = os.path.join(THIS_SCRIPT_DIR, "arch/eyeriss.yaml")

    with open(write_path, "w") as f:
        f.write("architecture:" + "\n")
        f.write("  version: 0.4" + "\n")
        f.write("  nodes:" + "\n")
        f.write("  - !Container" + "\n")
        f.write("    name: system" + "\n")
        f.write("    attributes:" + "\n")
        f.write("      technology: \"32nm\"" + "\n")
        f.write("      global_cycle_seconds: 1e-9" + "\n")
        f.write("\n")

        f.write("  - !Component" + "\n")
        f.write("    name: shared_glb" + "\n")
        f.write("    class: smartbuffer_SRAM" + "\n")
        f.write("    attributes:" + "\n")
        f.write("      depth: " + str(args.Global_Buffer_Depth) + "\n")
        f.write("      width: " + str(args.Global_Buffer_Width) + "\n")
        f.write("      n_banks: " + str(args.Global_Buffer_N_Banks) + "\n")
        f.write("      datawidth: 8" + "\n")
        f.write("      read_bandwidth: " + str(args.Global_Buffer_Read_Bandwidth) + "\n")
        f.write("      write_bandwidth: " + str(args.Global_Buffer_Write_Bandwidth) + "\n")
        f.write("    constraints:" + "\n")
        f.write("      temporal:" + "\n")
        f.write("        permutation: [C, S, R, Q, M, P, N]" + "\n")
        f.write("\n")

        f.write("  - !Container" + "\n")
        f.write("    name: PE_column" + "\n")
        f.write("    spatial: {meshX: " + str(args.meshX) + "}" + "\n")
        f.write("    constraints:" + "\n")
        f.write("      spatial:" + "\n")
        f.write("        permutation: [N, C, P, R, S, Q, M]" + "\n")
        f.write("        factors: [N=1, C=1, P=1, R=1, S=1, Q=16, P=1]" + "\n")
        # f.write("        factors: [N=1, C=1, P=1, R=1, S=1]" + "\n")
        f.write("        split: 8" + "\n")
        f.write("\n")

        f.write("  - !Container" + "\n")
        f.write("    name: PE" + "\n")
        f.write("    spatial: {meshY: " + str(args.meshY) + "}" + "\n")
        f.write("    constraints:" + "\n")
        f.write("      spatial:" + "\n")
        f.write("        split: 4" + "\n")
        f.write("        permutation: [N, P, Q, R, S, C, M]" + "\n")
        f.write("        factors: [N=1, P=1, Q=1, R=1, S=1, C=1, M=16]" + "\n")
        # f.write("        factors: [N=1, P=1, Q=1, R=1]" + "\n")
        f.write("\n")

        f.write("  - !Parallel" + "\n")
        f.write("    nodes:" + "\n")
        f.write("    - !Component" + "\n")
        f.write("      name: ifmap_spad" + "\n")
        f.write("      class: smartbuffer_RF" + "\n")
        f.write("      attributes:" + "\n")
        f.write("        depth: " + str(args.ifmap_spad_Depth) + "\n")
        f.write("        width: " + str(args.ifmap_spad_Width) + "\n")
        f.write("        datawidth: 8" + "\n")
        f.write("        read_bandwidth: 2" + "\n")
        f.write("        write_bandwidth: 2" + "\n")
        f.write("      constraints:" + "\n")
        # f.write("        dataspace: {keep: [Inputs], bypass: [Weights, Outputs]}" + "\n")
        f.write("        dataspace: {keep: [Inputs]}" + "\n")
        f.write("        temporal:" + "\n")
        f.write("          permutation: [N, M, C, P, Q, R, S]" + "\n")
        f.write("          factors: [N=1, M=1, C=1, P=1, Q=1, R=1, S=1]" + "\n")
        # f.write("          factors: [N=1, M=1, C=1, P=1, Q=1, R=1, S=1]" + "\n")
        f.write("\n")

        f.write("    - !Component" + "\n")
        f.write("      name: weights_spad" + "\n")
        f.write("      class: smartbuffer_RF" + "\n")
        f.write("      attributes:" + "\n")
        f.write("        depth: " + str(args.weights_spad_Depth) + "\n")
        f.write("        width: " + str(args.weights_spad_Width) + "\n")
        f.write("        datawidth: 8" + "\n")
        f.write("        read_bandwidth: 2" + "\n")
        f.write("        write_bandwidth: 2" + "\n")
        f.write("      constraints:" + "\n")
        # f.write("        dataspace: {keep: [Weights], bypass: [Inputs, Outputs]}" + "\n")
        f.write("        dataspace: {keep: [Weights]}" + "\n")
        f.write("        temporal:" + "\n")
        f.write("          permutation: [N, M, P, Q, S, C, R]" + "\n")
        f.write("          factors: [N=1, M=1, P=1, Q=1, S=1, C=1, R=1]" + "\n")
        # f.write("          factors: [N=1, M=1, P=1, Q=1, S=1]" + "\n")
        f.write("\n")

        f.write("    - !Component" + "\n")
        f.write("      name: psum_spad" + "\n")
        f.write("      class: smartbuffer_RF" + "\n")
        f.write("      attributes:" + "\n")
        f.write("        depth: " + str(args.ofmap_spad_Depth) + "\n")
        f.write("        width: " + str(args.ofmap_spad_Width) + "\n")
        f.write("        update_fifo_depth: 2" + "\n")
        f.write("        datawidth: 8" + "\n")
        f.write("        read_bandwidth: 2" + "\n")
        f.write("        write_bandwidth: 2" + "\n")
        f.write("      constraints:" + "\n")
        # f.write("        dataspace: {keep: [Outputs], bypass: [Inputs, Weights]}" + "\n")
        f.write("        dataspace: {keep: [Outputs]}" + "\n")
        f.write("        temporal:" + "\n")
        f.write("          permutation: [N, C, P, Q, R, S, M]" + "\n")
        f.write("          factors: [N=1, C=3, P=1, Q=1, R=1, S=1, M=1]" + "\n")
        # f.write("          factors: [N=1, C=1, R=1, S=1, P=1, Q=1]" + "\n")
        f.write("\n")

        f.write("  - !Component" + "\n")
        f.write("    name: mac" + "\n")
        f.write("    class: intmac" + "\n")
        f.write("    attributes:" + "\n")
        f.write("      multiplier_width: 8" + "\n")
        f.write("      adder_width: 8" + "\n")
        f.write("\n")




#######################################
#### function for running timeloop ####
#######################################
def generate_problem(args):
    if args.problem_type == "CONV":
        conv_problem(args)
    elif args.problem_type == "GEMM":
        print("GEMM problem is not implemented yet")
        exit()


def generate_arch(args):
    if args.arch_type == "eyeriss":
        eyeriss_arch(args)
    elif args.arch_type == "nvdla":
        print("NVDLA architecture is not implemented yet")
        exit()
    elif args.arch_type == "tpu":
        print("TPU architecture is not implemented yet")
        exit()
    else:
        print("Architecture type not supported yet!")
        exit()



def run_single(args):
    ## generating mapping
    spec = tl.Specification.from_yaml_files(
        os.path.join(THIS_SCRIPT_DIR, "timeloop_data/arch/" + args.arch_type + ".yaml"),
        os.path.join(THIS_SCRIPT_DIR, "timeloop_data/arch/components/*.yaml"),
        os.path.join(THIS_SCRIPT_DIR, "timeloop_data/prob/*.yaml"),
        os.path.join(THIS_SCRIPT_DIR, "timeloop_data/mapper/*.yaml"),
    )

    if args.clear:
        os.system("rm -rf "+os.path.join(THIS_SCRIPT_DIR, "timeloop_data/{}".format(args.output_dir)))
    else:
        try:
            tl.call_mapper(spec, output_dir=os.path.join(THIS_SCRIPT_DIR, "timeloop_data/{}".format(args.output_dir)))
            cycles, utilization = output_parser(args)
            return cycles, utilization
        except:
            return 999999999, 0



def output_parser(args):
    result_path = os.path.join(THIS_SCRIPT_DIR, "timeloop_data/{}/timeloop-mapper.stats.txt".format(args.output_dir))
    
    # if can read the file, implies the simulation is done
    # otherwise, the simulation fails, and return -1
    
    with open(result_path, "r") as f:
        lines = f.readlines()
        for line in reversed(lines):
            if line.startswith("Summary"):
                break
            if line.startswith("Cycles"):
                cycles = int(line.split(":")[1].strip())
            if line.startswith("Utilization"):
                utilization = float(line.split(":")[1].strip().replace("%", ""))
    return cycles, utilization




if __name__ == "__main__":

    #### testing the impact of depth of register files to the performance (ifmap, weights, ofmap)
    #### candidates is the power of 2 from 2^0 to 2^7 (1 to 128)
    #### record the results in a csv file

    #### problem size: H,W in {64, 128} and C_O in {16, 32, 64}

    cycles, utilization = run_single(args)
    

    