import os
import argparse
from ..util import *

THIS_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

import timeloopfe.v4 as tl



args = argparse.ArgumentParser(description="Problem Generator")
args.add_argument("--working_path", type=str)
args.add_argument("--layout_type", type=int, default=0)
args = args.parse_args()


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

    conv_problem(param_wl, args.working_path)
    eyeriss_arch(param_arch, args.working_path, args.layout_type)
    cycles, E_weights, E_ifmap, E_psum, E_glb = run_single()
    

    ## write the cost
    with open(args.working_path+'/costs.txt', 'w') as f:
        f.write(f"{cycles} {E_weights} {E_ifmap} {E_psum} {E_glb}\n")
    