import os
import timeloopfe.v4 as tl


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



class storage:
    def __init__(self, size, RCost, WCost, RBW, WBW):
        self.size = size
        self.RCost = RCost
        self.WCost = WCost
        self.RBW = RBW
        self.WBW = WBW

class bus:
    def __init__(self, BW, BusCost):
        self.BW = BW
        self.BusCost = BusCost

class macc:
    def __init__(self, meshX=16, meshY=16, MacCost=173.08):
        self.meshX = meshX
        self.meshY = meshY
        self.MacCost = MacCost

class EyeRiss:
    def __init__(self, al1, wl1, pl1, ul2, wbus, ibus, pbus, pes):
        self.al1 = al1
        self.wl1 = wl1
        self.pl1 = pl1
        self.ul2 = ul2
        self.wbus = wbus
        self.ibus = ibus
        self.pbus = pbus
        self.pes = pes


class Gemini_mapping:
    def __init__(self, H, W, C_O, C_I, K_H, K_W):
        self.H = H
        self.W = W
        self.C_O = C_O
        self.C_I = C_I
        self.K_H = K_H
        self.K_W = K_W



def run_Gemini_eval(Ct, max_c1, Kt, max_k1, Bt, filSize, ofmSize, 
                    k_reply, b_reply, phyarr_w, _al1_buf, core, args):

    C1 = min(Ct, max_c1)
    C2 = (Ct + C1 - 1) // C1
    K1 = min(Kt, max_k1)
    K2 = (Kt + K1 - 1) // K1
    comp_time = args.W * args.K_W * Kt * Ct * Bt

    # //for l2
    ul2_ww = 0
    ul2_wr = filSize
    ul2_pr = ul2_pw = ofmSize * C2
    ul2_ar = K2 * args.C_I * args.B
    ul2_aw = 0
    ul2_r = ul2_wr + ul2_ar + ul2_pr
    ul2_w = ul2_ww + ul2_aw + ul2_pw

    # // for interconnect
    bus_ww = ul2_wr
    bus_aw = ul2_ar
    bus_pw = ul2_pr + ul2_pw - ofmSize
    d_wbus = (bus_ww + core.pes.meshX * core.pes.meshY * 2 - 1) // (core.pes.meshX * core.pes.meshY * 2)
    d_ibus = (bus_aw + core.pes.meshX * core.pes.meshY * 2 - 1) // (core.pes.meshX * core.pes.meshY * 2)
    d_pbus = (bus_pw + core.pes.meshX * core.pes.meshY * 2 - 1) // (core.pes.meshX * core.pes.meshY * 2)

    # // for l1
    tot_op = args.W * args.H * args.C_I * args.C_O * args.K_W * args.K_H * args.B
    wl1_r = al1_r = tot_op
    pl1_r = pl1_w = tot_op - ofmSize
    wl1_w = filSize * b_reply * phyarr_w

    al1_k = K2 * k_reply
    if(Kt % K1 == 1):
        al1_k = al1_k - k_reply + (args.K_W % k_reply)
    al1_w = _al1_buf * al1_k

    ibuf_energy = (al1_w * core.al1.WCost + al1_r * core.al1.RCost) / tot_op
    wbuf_energy = (wl1_w * core.wl1.WCost + wl1_r * core.wl1.RCost) / tot_op
    pbuf_energy = (pl1_w * core.pl1.WCost + pl1_r * core.pl1.RCost) / tot_op
    ubuf_energy = (ul2_w * core.ul2.WCost + ul2_r * core.ul2.RCost) / tot_op
    noc_energy =  (bus_aw * core.ibus.BusCost + bus_ww * core.wbus.BusCost + bus_pw * core.pbus.BusCost) / tot_op
    mac_energy =  core.pes.MacCost
    
    tot_time = max(d_ibus, d_pbus, d_wbus, comp_time)

    return [mac_energy, wbuf_energy, ibuf_energy, pbuf_energy, ubuf_energy, noc_energy], tot_time




def run_Gemini(args, core):

    tot_ifmW = (args.W - 1) + args.K_W   # (args.W - 1) * args.S_W + args.K_W
    filSize = args.C_I * args.C_O * args.K_W * args.K_H
    ofmSize = args.C_O * args.B * args.W * args.H
    _al1_buf = args.H * args.C_I * args.H * args.B * tot_ifmW

    fold_w = (args.H + core.pes.meshX - 1) // core.pes.meshX
    phyarr_w = (args.H + fold_w - 1) // fold_w

    # fill x dimension (w axis) with H first, then C_O if possible
    k_reply_w = max(core.pes.meshX // args.H, 1)
    k_reply_w = min(args.C_O, k_reply_w)
    if args.H < core.pes.meshX:
        mapping_x = Gemini_mapping(args.H, 1, k_reply_w, 1, 1, 1)
    else:
        mapping_x = Gemini_mapping(core.pes.meshX, 1, 1, 1, 1, 1)
    reply_h = core.pes.meshY // args.K_H

    remain_k = (args.C_O + k_reply_w - 1) // k_reply_w

    max_c1 = core.al1.size // args.K_W
    max_k1 = core.wl1.size // (args.K_W * max_c1)
    total_b = args.B * fold_w

    max_kreplh = min( (remain_k + max_k1 - 1) // max_k1, reply_h)
    max_creplh = min( (args.C_I + max_c1 - 1) // max_c1, reply_h)

    if(max_creplh * max_kreplh >= reply_h):
        b_reply = 1
        Bt = total_b
        c_reply_h = reply_h // max_kreplh
        k_reply_h = reply_h // c_reply_h
        while True:
            Ct = (args.C_I + c_reply_h - 1) // c_reply_h
            Kt = (remain_k + k_reply_h - 1) // k_reply_h
            k_reply = k_reply_w * k_reply_h
            tot_energy, tot_time = run_Gemini_eval(Ct, max_c1, Kt, max_k1, Bt, filSize, ofmSize, 
                                                   k_reply, b_reply, phyarr_w, _al1_buf, core, args)
            if c_reply_h >= max_creplh:
                break
            k_reply_h = reply_h // (c_reply_h + 1)
            c_reply_h = reply_h // k_reply_h

    elif(max_creplh * max_kreplh * total_b >= reply_h):
        Kt = (remain_k + max_kreplh - 1) // max_kreplh
        k_reply = k_reply_w * max_kreplh
        rem_replh = reply_h // max_kreplh

        max_breplh = min(total_b, rem_replh)
        c_reply_h = rem_replh // max_breplh
        b_reply = rem_replh // c_reply_h

        while True:
            Ct = (args.C_I + c_reply_h - 1) // c_reply_h
            Bt = (total_b + b_reply - 1) // b_reply
            tot_energy, tot_time  = run_Gemini_eval(Ct, max_c1, Kt, max_k1, Bt, filSize, ofmSize,
                                                    k_reply, b_reply, phyarr_w, _al1_buf, core, args)
            if c_reply_h >= max_creplh:
                break
            b_reply = rem_replh // (c_reply_h + 1)
            c_reply_h = rem_replh // b_reply

        Ct = (args.C_I + max_creplh - 1) // max_creplh
        rem_replh = reply_h // max_creplh

        max_breplh = min(total_b, rem_replh)
        k_reply_h = rem_replh // max_breplh
        b_reply = rem_replh // k_reply_h
        while True:
            Kt = (remain_k + k_reply_h - 1) // k_reply_h
            Bt = (total_b + b_reply - 1) // b_reply
            k_reply = k_reply_w * k_reply_h
            tot_energy, tot_time = run_Gemini_eval(Ct, max_c1, Kt, max_k1, Bt, filSize, ofmSize,
                                                   k_reply, b_reply, phyarr_w, _al1_buf, core, args)
            if k_reply_h >= max_kreplh:
                break
            b_reply = rem_replh // (k_reply_h + 1)
            k_reply_h = rem_replh // b_reply
        
    else:
        Ct = (args.C_I + max_creplh - 1) // max_creplh
        Kt = (remain_k + max_kreplh - 1) // max_kreplh
        k_reply_h = max_kreplh
        c_reply_h = max_creplh
        k_reply = k_reply_w * max_kreplh
        b_reply = total_b
        Bt = 1
        assert Ct <= max_c1
        assert Kt <= max_k1
        
        tot_energy, tot_time = run_Gemini_eval(Ct, max_c1, Kt, max_k1, Bt, filSize, ofmSize,
                                               k_reply, b_reply, phyarr_w, _al1_buf, core, args)
    
    mapping_y = Gemini_mapping(b_reply, 1, k_reply_h, c_reply_h, args.K_H, 1)
    # print("Gemini PE X: H: {}, C_O: {}".format(mapping_x.H, mapping_x.C_O))
    # print("             Util: {:.2f}".format( mapping_x.H*mapping_x.C_O / args.meshX ))
    # print("          Y: H: {}, C_O: {}, C_I: {}".format(mapping_y.H, mapping_y.C_O, mapping_y.C_I))
    # print("             Util: {:.2f}".format( args.K_H*mapping_y.H*mapping_y.C_O*mapping_y.C_I / args.meshY ))

    return tot_energy, tot_time, (mapping_x, mapping_y)






#########################################
#### function for problem generation ####
#########################################

def conv_problem(args, param_wl, file_path):
    write_path = os.path.join(file_path, "prob/prob_conv.yaml")

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
def eyeriss_arch(args, param_arch, file_path, layout_ref_bank):

    ## generate eyeriss architecture
    write_path = os.path.join(file_path, "arch/eyeriss.yaml")

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
        f.write("      n_banks: " + str(args.Global_Buffer_N_Banks) + "\n")
        f.write("      datawidth: 8" + "\n")
        f.write("      read_bandwidth: "  + str(param_arch.Ubuf_width) + "\n")
        f.write("      write_bandwidth: " + str(param_arch.Ubuf_width) + "\n")
        f.write("    constraints:" + "\n")
        f.write("      temporal:" + "\n")
        if args.with_layout:
            layout_string = "        permutation: [" + layout_ref_bank[args.layout_type] + "]"
            if args.layout_type >= 0 and args.layout_type <= 5:
                f.write(layout_string + "\n")
            else:
                raise ValueError("Layout type not supported yet!")
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
        f.write("        depth: " + str(args.ifmap_spad_Depth) + "\n")
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
        f.write("        depth: " + str(args.weights_spad_Depth) + "\n")
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
        f.write("        depth: " + str(args.ofmap_spad_Depth) + "\n")
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




def output_parser(args, file_path):
    result_path = os.path.join(file_path, f"{args.output_dir}/timeloop-mapper.stats.txt")
    
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




def run_timeloop(args, file_path):
    spec = tl.Specification.from_yaml_files(
        os.path.join(file_path, "arch/" + args.arch_type + ".yaml"),
        os.path.join(file_path, "arch/components/*.yaml"),
        os.path.join(file_path, "prob/*.yaml"),
        os.path.join(file_path, "mapper/*.yaml"),
    )
    
    if args.clear:
        os.system("rm -rf "+os.path.join(file_path, f"{args.output_dir}"))
    else:
        os.system("rm -rf "+os.path.join(file_path, f"{args.output_dir}/timeloop-mapper.stats.txt"))  

        try:
            tl.call_mapper(spec, output_dir=os.path.join(file_path, f"{args.output_dir}"))
            energy, cycles, E_comp = output_parser(args)
            return energy, cycles, E_comp
        except:
            return -1, -1, -1