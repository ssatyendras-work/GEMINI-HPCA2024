import os
import argparse

THIS_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

import timeloopfe.v4 as tl



args = argparse.ArgumentParser(description="Problem Generator")
args.add_argument("--working_path", type=str)
args = args.parse_args()


class args_arch:
    def __init__(self, meshX=16, meshY=16, Ubuf_width=64, Ubuf_size=16384):
        self.meshX = meshX
        self.meshY = meshY
        self.Ubuf_width = Ubuf_width
        self.Ubuf_size = Ubuf_size


class args_wl:
    def __init__(self, B=1, W=224, H=224, C_I=3, K_W=3, K_H=3, C_O=64):
        self.B = B
        self.W = W
        self.H = H
        self.C_I = C_I
        self.K_W = K_W
        self.K_H = K_H
        self.C_O = C_O



#########################################
#### function for problem generation ####
#########################################

def conv_problem(param_wl):
    write_path = os.path.join(args.working_path, "prob_conv.yaml")

    with open(write_path, "w") as f:
        f.write("problem:" + "\n")
        f.write("  version: 0.4" + "\n")
        
        f.write("  instance:" + "\n")
        f.write("    N: " + str(param_wl.B) + "\n")
        f.write("    P: " + str(param_wl.W) + "\n")
        f.write("    Q: " + str(param_wl.H) + "\n")
        f.write("    C: " + str(param_wl.C_I) + "\n")
        f.write("    M: " + str(param_wl.C_O) + "\n")
        f.write("    R: " + str(param_wl.K_W) + "\n")
        f.write("    S: " + str(param_wl.K_H) + "\n")
        f.write("    Wstride: " + str(1) + "\n")
        f.write("    Hstride: " + str(1) + "\n")
        f.write("    Wdilation: " + str(1) + "\n")
        f.write("    Hdilation: " + str(1) + "\n")

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
def eyeriss_arch(param_arch):

    ## generate eyeriss architecture
    write_path = os.path.join(args.working_path, "eyeriss.yaml")

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
        f.write("      depth: " + str(param_arch.Ubuf_size // param_arch.Ubuf_width) + "\n")
        f.write("      width: " + str(param_arch.Ubuf_width * 8) + "\n")
        f.write("      n_banks: 32" + "\n")
        f.write("      datawidth: 8" + "\n")
        f.write("      read_bandwidth: "  + str(param_arch.Ubuf_width) + "\n")
        f.write("      write_bandwidth: " + str(param_arch.Ubuf_width) + "\n")
        f.write("    constraints:" + "\n")
        f.write("      temporal:" + "\n")
        f.write("        permutation: [C, P, Q]" + "\n")
        f.write("\n")


        f.write("  - !Container" + "\n")
        f.write("    name: PE_column" + "\n")
        f.write("    spatial: {meshX: " + str(param_arch.meshX) + "}" + "\n")
        f.write("    constraints:" + "\n")
        f.write("      spatial:" + "\n")
        f.write("        permutation: [N, C, P, R, S, Q, M]" + "\n")
        f.write("        factors: [C=1, P=1, R=1, S=1]" + "\n")
        f.write("        split: 8" + "\n")
        f.write("\n")

        f.write("  - !Container" + "\n")
        f.write("    name: PE" + "\n")
        f.write("    spatial: {meshY: " + str(param_arch.meshY) + "}" + "\n")
        f.write("    constraints:" + "\n")
        f.write("      spatial:" + "\n")
        f.write("        permutation: [N, P, Q, R, S, C, M]" + "\n")
        f.write("        factors: [N=1, P=1, Q=1, R=1]" + "\n")
        f.write("        split: 4" + "\n")
        f.write("\n")

        f.write("  - !Parallel" + "\n")
        f.write("    nodes:" + "\n")
        f.write("    - !Component" + "\n")
        f.write("      name: ifmap_spad" + "\n")
        f.write("      class: smartbuffer_RF" + "\n")
        f.write("      attributes:" + "\n")
        f.write("        depth: 4" + "\n")
        f.write("        width: 8" + "\n")
        f.write("        datawidth: 8" + "\n")
        f.write("        read_bandwidth: 2" + "\n")
        f.write("        write_bandwidth: 2" + "\n")
        f.write("      constraints:" + "\n")
        f.write("        dataspace: {keep: [Inputs], bypass: [Weights, Outputs]}" + "\n")
        f.write("        temporal:" + "\n")
        f.write("          permutation: [N, M, C, P, Q, R, S]" + "\n")
        f.write("          factors: [N=1, M=1, C=1, P=1, Q=1, R=1, S=1]" + "\n")
        f.write("\n")

        f.write("    - !Component" + "\n")
        f.write("      name: weights_spad" + "\n")
        f.write("      class: smartbuffer_RF" + "\n")
        f.write("      attributes:" + "\n")
        f.write("        depth: 4" + "\n")
        f.write("        width: 8" + "\n")
        f.write("        datawidth: 8" + "\n")
        f.write("        read_bandwidth: 2" + "\n")
        f.write("        write_bandwidth: 2" + "\n")
        f.write("      constraints:" + "\n")
        f.write("        dataspace: {keep: [Weights], bypass: [Inputs, Outputs]}" + "\n")
        f.write("        temporal:" + "\n")
        f.write("          permutation: [N, M, P, Q, S, C, R]" + "\n")
        f.write("          factors: [N=1, M=1, P=1, Q=1, S=1]" + "\n")
        f.write("\n")

        f.write("    - !Component" + "\n")
        f.write("      name: psum_spad" + "\n")
        f.write("      class: smartbuffer_RF" + "\n")
        f.write("      attributes:" + "\n")
        f.write("        depth: 4" + "\n")
        f.write("        width: 8" + "\n")
        f.write("        update_fifo_depth: 2" + "\n")
        f.write("        datawidth: 8" + "\n")
        f.write("        read_bandwidth: 2" + "\n")
        f.write("        write_bandwidth: 2" + "\n")
        f.write("      constraints:" + "\n")
        f.write("        dataspace: {keep: [Outputs], bypass: [Inputs, Weights]}" + "\n")
        f.write("        temporal:" + "\n")
        f.write("          permutation: [N, C, P, Q, R, S, M]" + "\n")
        f.write("          factors: [C=1, R=1, S=1, P=1, Q=1]" + "\n")
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

def output_parser():
    result_path = os.path.join(args.working_path, "output/timeloop-mapper.stats.txt")
    
    # if can read the file, implies the simulation is done
    # otherwise, the simulation fails, and return -1
    
    ## read by different cases
    ## case 0: not started
    ## case 1: L2 buffer
    ##         -- target: energy cost per action
    ## case 2: temporal stop
    ## case 3: Summary
    ##         -- target: energy (L2 buffer, L1 buffers, compute), cycles
    case = 0
    with open(result_path, "r") as f:
        lines = f.readlines()

        for line in lines:
            ## detect the case
            if (case==0) and (line.startswith("===")):
                if line.split()[1] == "shared_glb":
                    case = 1
            elif (case==2) and (line.startswith("Summary")):
                case = 3

            ## case 1: find the energy cost per action of L2 buffer
            if case == 1:
                line = line.strip()
                if line.startswith("Vector access energy"):
                    case = 2

            ## case 2: find the energy and cycles
            elif case == 3:
                line = line.strip()
                if line.startswith("Cycles"):
                    cycles = int(line.split(":")[1].strip())
                elif line.startswith("Energy"):
                    energy = float(line.split(":")[1].split()[0])
                    unit = line.split(":")[1].split()[1]
                    if unit == "pJ":
                        pass
                    elif unit == "nJ":
                        energy = energy * 1e3
                    elif unit == "uJ":
                        energy = energy * 1e6
                    elif unit == "mJ":
                        energy = energy * 1e9
                    elif unit == "J":
                        energy = energy * 1e12
                    else:
                        print("Energy unit not supported yet!")
                        exit()
                elif line.startswith("mac"):
                    E_mac = float(line.split("=")[1].strip())
                elif line.startswith("weights_spad"):
                    E_weights = float(line.split("=")[1].strip())
                elif line.startswith("ifmap_spad"):
                    E_ifmap = float(line.split("=")[1].strip())
                elif line.startswith("psum_spad"):
                    E_psum = float(line.split("=")[1].strip())
                elif line.startswith("shared_glb"):
                    E_glb = float(line.split("=")[1].strip())

    
    E_comp = [E_mac, E_weights, E_ifmap, E_psum, E_glb]
    return energy, cycles, E_comp





def run_single():
    ## generating mapping
    spec = tl.Specification.from_yaml_files(
        os.path.join(args.working_path, "*.yaml"),
        os.path.join(THIS_SCRIPT_DIR, "arch/components/*.yaml"),
        os.path.join(THIS_SCRIPT_DIR, "mapper/*.yaml"),
    )

    os.system("rm -rf "+os.path.join(args.working_path, "output/timeloop-mapper.stats.txt"))
    try:
        tl.call_mapper(spec, output_dir=os.path.join(args.working_path, "output"))
        _, cycles, E_comp = output_parser()
        return cycles, E_comp[1], E_comp[2], E_comp[3], E_comp[4]
    except:
        return 999999999, 999999999, 999999999, 999999999, 999999999



if __name__ == "__main__":

    ## read workload params
    with open(args.working_path+'/prob_data.txt', 'r') as f:
        line = f.readline()
        # Split by spaces and convert to int (or float, if you need)
        Gemini_wl_params = [int(x) for x in line.strip().split()]
        # Unpack if you want
        B, W, H, C, K, S, R, sW, sH = Gemini_wl_params
    
    
    with open(args.working_path+'/arch_data.txt', 'r') as f:
        line = f.readline()
        Gemini_arch_params = [int(x) for x in line.strip().split()]
        # Unpack
        ul2_Size, Xarray, Yarray = Gemini_arch_params
    
    param_wl = args_wl(B=B, W=W, H=H, C_I=C, C_O=K, K_W=R, K_H=S)
    param_arch = args_arch(Ubuf_size=ul2_Size, meshX=Xarray, meshY=Yarray)

    conv_problem(param_wl)
    eyeriss_arch(param_arch)
    cycles, E_weights, E_ifmap, E_psum, E_glb = run_single()
    

    ## write the cost
    with open(args.working_path+'/costs.txt', 'w') as f:
        f.write(f"{cycles} {E_weights} {E_ifmap} {E_psum} {E_glb}\n")
    