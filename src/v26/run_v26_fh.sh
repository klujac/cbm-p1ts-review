#!/usr/bin/env bash
# run_v26_fh.sh VARIANT TASKS.csv [NSH]  -- frozen-hyper-parameter ablation; NSH processes alternating GPU 0/1; resumable.
set -u
V=$1; T=$2; NSH=${3:-8}
: "${DRIVER_DIR:?source ~/INS26_rewizja/env.sh first}"; : "${OLD_RUN:?source ~/INS26_rewizja/env.sh first}"
mkdir -p logs_v26
pids=()
for i in $(seq 0 $((NSH-1))); do
  CUDA_VISIBLE_DEVICES=$((i%2)) nohup python run_v26_fh.py "$V" "$T" "$i" "$NSH" > "logs_v26/${V}_${i}.log" 2>&1 &
  pids+=($!)
done
echo "started ${NSH} shards of ${V}: ${pids[*]}"
wait "${pids[@]}"
echo "finished ${V}: $(ls results_v26/${V}/s*_f*.json 2>/dev/null | grep -vc failed) done, $(ls results_v26/${V}/*.failed.json 2>/dev/null | wc -l) failed"
