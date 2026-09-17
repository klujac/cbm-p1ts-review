#!/usr/bin/env bash
# run_v26_large.sh ALG TASKS.csv [NSH] -- large-scale track; GPU algs alternate GPU 0/1; resumable.
set -u
A=$1; T=$2; NSH=${3:-6}
: "${DRIVER_DIR:?source ~/INS26_rewizja/env.sh first}"
mkdir -p logs_v26
pids=()
for i in $(seq 0 $((NSH-1))); do
  CUDA_VISIBLE_DEVICES=$((i%2)) nohup python run_v26_large.py "$A" "$T" "$i" "$NSH" > "logs_v26/large_${A}_${i}.log" 2>&1 &
  pids+=($!)
done
echo "started ${NSH} shards of large/${A}: ${pids[*]}"
wait "${pids[@]}"
echo "finished large/${A}: $(ls results_v26/large/s42_*_${A}_f*.json 2>/dev/null | grep -vc failed) done"
