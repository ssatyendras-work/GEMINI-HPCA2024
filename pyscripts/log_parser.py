
# Parses a chiplet accelerator log file and returns a DataFrame of layer-level metrics.
# Parameters:
#     file_path (str): Path to the log file 
# Returns:
#     pd.DataFrame: Parsed metrics for each layer block

import re
import pandas as pd

def parse_chiplet_log(file_path):
    
    with open(file_path, "r") as file:
        log_lines = file.readlines()

    top_data_all = []

    layer_data_preSPM = []
    sub_layer_data_preSPM = []

    layer_data_postSPM = []
    sub_layer_data_postSPM = []

    Top_pattern = re.compile(
        r'T\s+\d+/\d+\s+E:([\deE+.-]+),\s+T:(\d+),\s+Cost:([\deE+.-]+)\s+'
        r'Ubuf/Buf/Bus/Mac:([\deE+.-]+)/([\deE+.-]+)/([\deE+.-]+)/([\deE+.-]+)\s+'
        r'NoC energy = ([\deE+.-]+)\s+NoP energy = ([\deE+.-]+)\s+DRAM energy = ([\deE+.-]+)\s+'
        r'NoC\(NoC_tot_hops= (\d+)\s+NoP_tot_hops= (\d+),core_hops=(\d+), DRAM acc=(\d+)\)\s+'
        r'NoCtime = (\d+)\s+NocBW = (\d+)'
    )

    LG_pattern = re.compile(
        r'S\s+\d+/\d+\s+E:([\deE+.-]+),\s+T:(\d+),\s+Cost:([\deE+.-]+)\s+'
        r'Ubuf/Buf/Bus/Mac:([\deE+.-]+)/([\deE+.-]+)/([\deE+.-]+)/([\deE+.-]+)\s+'
        r'NoC energy = ([\deE+.-]+)\s+NoP energy = ([\deE+.-]+)\s+DRAM energy = ([\deE+.-]+)\s+'
        r'NoC\(NoC_tot_hops= (\d+)\s+NoP_tot_hops= (\d+),core_hops=(\d+), DRAM acc=(\d+)\)\s+'
        r'NoCtime = (\d+)\s+NocBW = (\d+)'
    )
    sub_LG_pattern = re.compile(
        r'\s*(\w[\w\d_]+)\s+1\s+\(([^)]+)\)\s+util:([\d.]+)/([\d.]+)\s+E:([\deE+.+-]+),\s+T:(\d+),\s+Cost:([\deE+.+-]+)\s+'
        r'Ubuf/Buf/Bus/Mac:([\deE+.+-]+)/([\deE+.+-]+)/([\deE+.+-]+)/([\deE+.+-]+).*?'
        r'NoC energy = ([\deE+.+-]+).*?DRAM energy = ([\deE+.+-]+).*?NoC\(NoC_tot_hops= (\d+)[^,]*,.*?DRAM acc=(\d+).*?'
        r'Computing_time = (\d+)'
    )

    counter = 0
    postSPM = False
    for line in log_lines:
        match_Top    = Top_pattern.search(line)
        match_LG     = LG_pattern.search(line)
        match_sub_LG = sub_LG_pattern.search(line)

        ## check if is before SPM or not
        line = line.strip()
        if line.startswith("Schtree_spm (after SPM search):"):
            counter = 0
            postSPM = True

        if match_Top:
            Top_data = {
                "energy":       float(match_Top.group(1)),
                "latency":      int(  match_Top.group(2)),
                "cost":         float(match_Top.group(3)),
                "ubuf":         float(match_Top.group(4)),
                "buf":          float(match_Top.group(5)),
                "bus":          float(match_Top.group(6)),
                "mac":          float(match_Top.group(7)),
                "NoC_energy":   float(match_Top.group(8)),
                "NoP_energy":   float(match_Top.group(9)),
                "DRAM_energy":  float(match_Top.group(10)),
                "NoC_tot_hops": int(  match_Top.group(11)),
                "NoP_tot_hops": int(  match_Top.group(12)),
                "core_hops":    int(  match_Top.group(13)),
                "DRAM_acc":     int(  match_Top.group(14)),
                "NoCtime":      int(  match_Top.group(15)),
                "NoC_BW":       int(  match_Top.group(16)),
                "LG_iter":      0
            }
            top_data_all.append(Top_data)

        elif match_LG:
            counter = counter+1
            LG_data = {
                "energy":       float(match_LG.group(1)),
                "latency":      int(  match_LG.group(2)),
                "cost":         float(match_LG.group(3)),
                "ubuf":         float(match_LG.group(4)),
                "buf":          float(match_LG.group(5)),
                "bus":          float(match_LG.group(6)),
                "mac":          float(match_LG.group(7)),
                "NoC_energy":   float(match_LG.group(8)),
                "NoP_energy":   float(match_LG.group(9)),
                "DRAM_energy":  float(match_LG.group(10)),
                "NoC_tot_hops": int(  match_LG.group(11)),
                "NoP_tot_hops": int(  match_LG.group(12)),
                "core_hops":    int(  match_LG.group(13)),
                "DRAM_acc":     int(  match_LG.group(14)),
                "NoCtime":      int(  match_LG.group(15)),
                "NoC_BW":       int(  match_LG.group(16)),
                "LG_iter":      int(  counter)
            }
            if postSPM:
                layer_data_postSPM.append(LG_data)
            else:
                layer_data_preSPM.append(LG_data)

        elif match_sub_LG:
            sub_LG_data = {
                "layer": match_sub_LG.group(1),
                "params": match_sub_LG.group(2),
                "util_actual":      float(match_sub_LG.group(3)),
                "util_peak":        float(match_sub_LG.group(4)),
                "energy":           float(match_sub_LG.group(5)),
                "latency":          int(  match_sub_LG.group(6)),
                "cost":             float(match_sub_LG.group(7)),
                "ubuf":             float(match_sub_LG.group(8)),
                "buf":              float(match_sub_LG.group(9)),
                "bus":              float(match_sub_LG.group(10)),
                "mac":              float(match_sub_LG.group(11)),
                "NoC_energy":       float(match_sub_LG.group(12)),
                "DRAM_energy":      float(match_sub_LG.group(13)),
                "NoC_hops":         int(  match_sub_LG.group(14)),
                "DRAM_acc":         int(  match_sub_LG.group(15)),
                "Computing_time":   int(  match_sub_LG.group(16)),
                "LG_iter":          int(  counter)
            }
            if postSPM:
                sub_layer_data_postSPM.append(sub_LG_data)
            else:
                sub_layer_data_preSPM.append(sub_LG_data)

    preSPM_data = {
        'top':   top_data_all[0], 
        'LG':    pd.DataFrame(layer_data_preSPM),
        'layer': pd.DataFrame(sub_layer_data_preSPM)
    }

    postSPM_data = {
        'top':   top_data_all[1], 
        'LG':    pd.DataFrame(layer_data_postSPM),
        'layer': pd.DataFrame(sub_layer_data_postSPM)
    }

    return preSPM_data, postSPM_data
