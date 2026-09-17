#!/usr/bin/env bash
# run_tabicl.sh TASKS.csv [NSH] -- TabICLv2 on GPU 0 only (GPU 1 reserved for another user); resumable.
set -u
T=$1; N=${2:-3}
mkdir -p logs_v26
pids=()
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=0 nohup ~/INS26_rewizja/venv_new/bin/python run_tabicl.py "$T" "$i" "$N" > "logs_v26/tabicl_${i}.log" 2>&1 &
  pids+=($!)
done
echo "started ${N} shards of TabICLv2: ${pids[*]}"
wait "${pids[@]}"
echo "finished: $(ls results_v26/tabicl/s*_f*.json 2>/dev/null | grep -vc failed) done, $(ls results_v26/tabicl/*.failed.json 2>/dev/null | wc -l) failed"
