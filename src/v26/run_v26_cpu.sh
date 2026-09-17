#!/usr/bin/env bash
# run_v26_cpu.sh TASKS.csv [NPROC]  -- CPU only, low priority (nice 10), one thread per process; resumable.
set -u
T=$1; N=${2:-16}
: "${DRIVER_DIR:?source ~/INS26_rewizja/env.sh first}"
mkdir -p logs_v26
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
pids=()
for i in $(seq 0 $((N-1))); do
  nohup nice -n 10 python run_v26_cpu.py "$T" "$i" "$N" > "logs_v26/tuned6_${i}.log" 2>&1 &
  pids+=($!)
done
echo "started ${N} CPU shards: ${pids[*]}"
wait "${pids[@]}"
echo "finished: $(ls results_v26/tuned6/s*_f*.json 2>/dev/null | grep -vc failed) done, $(ls results_v26/tuned6/*.failed.json 2>/dev/null | wc -l) failed"
