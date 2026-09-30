#!/usr/bin/env bash
# ================================================================
# Eval the performance by Timeloop (parallel runner)
#
# Usage:
#   bash eval.sh <param_file_base>
# Example:
#   bash eval.sh eyeriss_resnet50_tops128   # reads eyeriss_resnet50_tops128.log
#
# Input file format (16 fields per line):
#   mm nn xx yy ss bb rr ff xcut ycut _serdes_lane _DRAM_bw _NoC_bw _mac_num _ul3 TOPS
#
# search() call contract (21 args total):
#   search $1..$16  $TL_workspace  result.log  out.log  err.log  search.log
#            ^^^^^^  ($17)         ($18)       ($19)    ($20)    ($21)
#
# Concurrency:
#   export MAX_JOBS=32            # default 32
#   export OMP_NUM_THREADS=1      # recommended if stschedule is multithreaded
# ================================================================

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <param_file_base>" >&2
  exit 1
fi

PARAM_BASE="$1"
PARAM_FILE="${PARAM_BASE}.log"

if [[ ! -f "$PARAM_FILE" ]]; then
  echo "[ERROR] File not found: $PARAM_FILE" >&2
  exit 1
fi

# ---------- Output root ----------
output_path="$(pwd)/dse_log/${PARAM_BASE}"
mkdir -p "$output_path"

# ---------- Filenames & constants ----------
output_point_file="result.log"
output_struct_file="out.log"
err_log="err.log"
search_log="search.log"

# Network names (index = nn)
network=(darknet19 vgg resnet50 googlenet resnet101 densenet ires gnmt lstm zfnet transformer transformer_cell pnasnet resnext50 resnet152 vgg_tiny)

# Summary depth
max_dir_depth=16

# ---------- Concurrency ----------
MAX_JOBS="${MAX_JOBS:=32}"
active=0
line_no=0

# Limit threads of internal libraries to avoid oversubscription
: "${OMP_NUM_THREADS:=1}"
export OMP_NUM_THREADS

# ---------- search() wrapper ----------
# NOTE: MUST return 0 even if elapsed seconds > 0 (to avoid set -e aborts)
search(){
    start=$(date +%s)
    # $1..$16 = 16 params; $17 = TL workspace path (echoed into stdin)
    # pass $18 to stschedule as its CLI arg; stdout-> $19, stderr-> $20
    echo "$1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17}" \
      | ./build/stschedule ${18} > ${19} 2> ${20}
    end=$(date +%s)
    iter_time=$(( end - start ))
    # Append a timing line to $21 (search.log)
    printf "INPUT(%1d %2d %3d %3d %3d %3d %3d %2d %3d %3d %3d %8d %3d %4d %4d %8d) ITERATION_TIME(%5d)seconds\n" \
      $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} ${16} $iter_time >> ${21}
    return 0
}

# ---------- Per-line job (safe for backgrounding) ----------
run_one() {
  # params in the same order as the line
  local mm="$1" nn="$2" xx="$3" yy="$4" ss="$5" bb="$6" rr="$7" ff="$8" \
        xcut="$9" ycut="${10}" _serdes_lane="${11}" _DRAM_bw="${12}" \
        _NoC_bw="${13}" _mac_num="${14}" _ul3="${15}" TOPS="${16}"

  # Recreate the SAME hierarchy
  local temp_path0="$output_path/${network[$nn]}"
  local temp_path1="$temp_path0/DRAM_bw_${_DRAM_bw}"
  local temp_path2="$temp_path1/NoC_bw_${_NoC_bw}"
  local temp_path3="$temp_path2/ul3_${_ul3}"
  local temp_path4="$temp_path3/NoP_bw_${_serdes_lane}"
  local temp_path5="$temp_path4/mac_num_${_mac_num}"
  local temp_path_innerest="$temp_path5/xcut_${xcut}_ycut_${ycut}"
  local tl_ws="$temp_path_innerest/TL"

  mkdir -p "$tl_ws" "$temp_path_innerest"

  # Prepare files (unique per job)
  : > "$temp_path_innerest/$search_log"
  : > "$temp_path_innerest/$err_log"
  : > "$temp_path_innerest/$output_point_file"
  : > "$temp_path_innerest/$output_struct_file"

  # Optional partition sanity checks (uncomment if you want to skip invalid combos)
  # if (( xx % xcut != 0 || yy % ycut != 0 )); then
  #   echo "[INFO] Skip invalid partition: xx% xcut or yy% ycut != 0  (nn=$nn xcut=$xcut ycut=$ycut)"
  #   return 0
  # fi
  # if (( xcut * ycut == 1 && _serdes_lane != _NoC_bw )); then
  #   echo "[INFO] Skip 1x1 chiplet with mismatched NoP/NoC"
  #   return 0
  # fi

  # Invoke search with 16 params + TL workspace + 4 files
  search "$mm" "$nn" "$xx" "$yy" "$ss" "$bb" "$rr" "$ff" "$xcut" "$ycut" \
         "$_serdes_lane" "$_DRAM_bw" "$_NoC_bw" "$_mac_num" "$_ul3" "$TOPS" \
         "$tl_ws" \
         "$temp_path_innerest/$output_point_file" \
         "$temp_path_innerest/$output_struct_file" \
         "$temp_path_innerest/$err_log" \
         "$temp_path_innerest/$search_log"
}

echo "******************** 128TOPS DSE (eval) Start ********************"
echo "[INFO] Input file: $PARAM_FILE"
echo "[INFO] Output dir: $output_path"
echo "[INFO] MAX_JOBS=${MAX_JOBS}  (override with: export MAX_JOBS=<n>)"

# ---------- Main loop (parallel) ----------
# Each non-empty, non-comment line must have exactly 16 fields
while IFS=$' \t' read -r \
  mm nn xx yy ss bb rr ff xcut ycut _serdes_lane _DRAM_bw _NoC_bw _mac_num _ul3 TOPS extra || [[ -n "${mm-}" ]]
do

  # Skip blanks/comments
  [[ -z "${mm-}" ]] && continue
  [[ "${mm}" =~ ^# ]] && continue

  ((++line_no))

  # Enforce exactly 16 fields
  if [[ -n "${extra-}" ]]; then
    echo "[WARN] Skipping line with >16 fields (line $line_no)" >&2
    continue
  fi


  # Validate nn index
  if (( nn < 0 || nn >= ${#network[@]} )); then
    echo "[WARN] nn=$nn out of range (0..$(( ${#network[@]} -1 ))), skipping line $line_no." >&2
    continue
  fi

  # Launch one job in background
  run_one "$mm" "$nn" "$xx" "$yy" "$ss" "$bb" "$rr" "$ff" "$xcut" "$ycut" \
          "$_serdes_lane" "$_DRAM_bw" "$_NoC_bw" "$_mac_num" "$_ul3" "$TOPS" &

  ((++active))
  
  if (( active >= MAX_JOBS )); then
    # wait for any one to finish (don't abort if a job exits non-zero elsewhere)
    wait -n || true
    ((--active))
  fi
done < "$PARAM_FILE"

# Wait for remaining jobs
wait

echo "[INFO] Finished $line_no lines."
echo "[INFO] Summarizing experiment results..."
bash ./summary.sh "$output_path" "$max_dir_depth" || true
echo "[INFO] Summary at $output_path/result.xlsx"
echo "[INFO] Summary accomplished."
echo "[INFO] Best Arch found."
python3 pyscripts/best_arch.py "$output_path/result.csv" "$output_path/best_arch.txt" || true
echo "******************** 128TOPS DSE (eval) Finish *******************"
