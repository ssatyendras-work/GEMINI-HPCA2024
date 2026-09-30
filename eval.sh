# Eval the performance by Timeloop

# Usage: search $mm $nn $xx $yy $ss $bb $rr $ff $xcut $ycut $_serdes_lane $_DRAM_bw $_NoC_bw $_mac_num $_ul3 $TOPS $TL_workspace result.log out.log err.log search.log
# mm: core type (0: Polar (NVDLA), 1: Eyeriss))
# nn: network type (0: darknet19, 1: vgg, 2: resnet50, 3: googlenet, 4: resnet101, 5: densenet, 6: ires, 7: gnmt, 8: lstm, 9: zfnet, 10: transformer, 11: transformer_cell, 12: pnasnet, 13: resnext50, 14: resnet152)
# xx & yy: means the number of cores in the x and y axis, respectively. Default noc topology is mesh.
# ss: means stride, whose definition can be found in ISCA2023 SET, HPCA2024GEMINI, ASPLOS2019TANGRAM.
# bb: means batch size (2^n is recommanded).
# rr: means exploration rounds number, which will be multiplied with a coefficient.
# ff: means optimization goal, concrete formula can be found in main.cpp.
# xcut & ycut: means the chiplet partition granularity at the x and y dimension.
# _serdes_lane: number of serdes lanes
# _DRAM_bw: DRAM bandwidth
# _NoC_bw: NoC bandwidth
# _mac_num: number of MACs
# _ul3: size of ul3
# TOPS: total TOPS

# read all the candidates from list.txt



# Usage: bash eval.sh <param_file>
# Reads from <param_file>.log
# Writes outputs into ./dse_log/<param_file>/

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <param_file>" >&2
  exit 1
fi

PARAM_BASE="$1"
PARAM_FILE="${PARAM_BASE}.log"

if [[ ! -f "$PARAM_FILE" ]]; then
  echo "File not found: $PARAM_FILE" >&2
  exit 1
fi

output_path="$(pwd)/dse_log/${PARAM_BASE}"
mkdir -p "$output_path"


# *************************** Output File ****************************
output_point_file="result.log"
output_struct_file="out.log"
err_log="err.log"
search_log="search.log"



# *********************** search() wrapper ************************
# search $1 ... $16 TOPS result.log out.log err.log search.log

search(){
    start=$(date +%s)
    echo "$1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17}" | ./build/stschedule ${18} > ${19} 2> ${20}
    end=$(date +%s)
    iter_time=$(( end - start ))
    printf "INPUT(%1d %2d %3d %3d %3d %3d %3d %2d %3d %3d %3d %8d %3d %4d %4d %8d) ITERATION_TIME(%5d)seconds\n" $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} $iter_time >> ${21}
    return $iter_time
}


# *********************** Global constants ***********************
TOPS=`expr 128 \* 1024`
network=(darknet19 vgg resnet50 googlenet resnet101 densenet ires gnmt lstm zfnet transformer transformer_cell pnasnet resnext50 resnet152 vgg_tiny)
max_dir_depth=16


# Read exactly 16 fields per non-empty, non-comment line
while IFS=$' \t' read -r \
    mm nn xx yy ss bb rr ff xcut ycut _serdes_lane _DRAM_bw _NoC_bw _mac_num _ul3 TOPS extra || [[ -n "${mm-}" ]]
do
    # skip blanks or comments
    [[ -z "${mm-}" ]] && continue
    [[ "${mm}" =~ ^# ]] && continue
    
    # enforce exactly 16 fields
    if [[ -n "${extra-}" ]]; then
        echo "[WARN] Skipping line with >16 fields: $mm $nn ... (line $((line_no+1)))" >&2
        continue
    fi
    

    # Optional: validate nn within bounds
    if (( nn < 0 || nn >= ${#network[@]} )); then
        echo "[WARN] nn=$nn out of range (0..$(( ${#network[@]} -1 ))), skipping line $((line_no+1)))." >&2
        continue
    fi
    

    # ----- Recreate the SAME path hierarchy -----
    temp_path0="$output_path/${network[$nn]}"
    temp_path1="$temp_path0/DRAM_bw_${_DRAM_bw}"
    temp_path2="$temp_path1/NoC_bw_${_NoC_bw}"
    temp_path3="$temp_path2/ul3_${_ul3}"
    temp_path4="$temp_path3/NoP_bw_${_serdes_lane}"
    temp_path5="$temp_path4/mac_num_${_mac_num}"
    temp_path_innerest="$temp_path5/xcut_${xcut}_ycut_${ycut}"
    mkdir -p "$temp_path_innerest"
    mkdir $temp_path_innerest/TL

    # Prepare files
    : > "$temp_path_innerest/$search_log"
    : > "$temp_path_innerest/$err_log"
    : > "$temp_path_innerest/$output_point_file"
    : > "$temp_path_innerest/$output_struct_file"

    # echo "[INFO] Simulating ${network[$nn]} DRAM=${_DRAM_bw} NoC=${_NoC_bw} ul3=${_ul3} NoP=${_serdes_lane} MAC=${_mac_num} xcut=${xcut} ycut=${ycut}"

    # ----- Call search with exactly the 16 params from the file -----
    search "$mm" "$nn" "$xx" "$yy" "$ss" "$bb" "$rr" "$ff" "$xcut" "$ycut" "$_serdes_lane" "$_DRAM_bw" "$_NoC_bw" "$_mac_num" "$_ul3" "$TOPS" \
           "$temp_path_innerest/TL" \
           "$temp_path_innerest/$output_point_file" \
           "$temp_path_innerest/$output_struct_file" \
           "$temp_path_innerest/$err_log" \
           "$temp_path_innerest/$search_log"

done < "$PARAM_FILE"

echo "[INFO] Finished $line_no lines."
echo "[INFO] Summarizing experiment results..."
bash ./summary.sh "$output_path" "$max_dir_depth"
echo "[INFO] Summary at $output_path/result.xlsx"
echo "[INFO] Summary accomplished."
echo "[INFO] Best Arch found."
python3 pyscripts/best_arch.py "$output_path/result.csv" "$output_path/best_arch.txt"
echo "******************** 128TOPS DSE (eval) Finish *******************"
