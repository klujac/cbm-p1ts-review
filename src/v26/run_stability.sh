#!/usr/bin/env bash
# run_stability.sh ALG TASKS.csv [NSH] -- GPU 0 only (GPU 1 reserved); resumable.
set -u
A=$1; T=$2; N=${3:-4}
: "${DRIVER_DIR:?source ~/INS26_rewizja/env.sh first}"
mkdir -p logs_v26
pids=()
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=0 nohup python run_stability.py "$A" "$T" "$i" "$N" > "logs_v26/stab_${A}_${i}.log" 2>&1 &
  pids+=($!)
done
echo "started ${N} shards of stability/${A}: ${pids[*]}"
wait "${pids[@]}"
echo "finished stability/${A}: $(ls results_v26/stability/*_${A}_f*.json 2>/dev/null | grep -vc failed) done"
