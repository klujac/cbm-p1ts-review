#!/usr/bin/env bash
# smoke_fh.sh -- one process on GPU 0, a few variants on two tasks (diabetes CBM s42 f0; magic CBM_KE s43 f6)
set -u
: "${DRIVER_DIR:?source ~/INS26_rewizja/env.sh first}"; : "${OLD_RUN:?source ~/INS26_rewizja/env.sh first}"
mkdir -p logs_v26
for V in fh_ref_v25 fh_clamp fh_norules fh_fixedppl; do
  CUDA_VISIBLE_DEVICES=0 python run_v26_fh.py "$V" tasks_fh_smoke_cbm.csv 0 1 > "logs_v26/smoke_${V}.log" 2>&1
  grep -h "\[V26-FH\].* f0 " "logs_v26/smoke_${V}.log"
done
for V in fh_nokl fh_thard; do
  CUDA_VISIBLE_DEVICES=0 python run_v26_fh.py "$V" tasks_fh_smoke_ke.csv 0 1 > "logs_v26/smoke_${V}.log" 2>&1
  grep -h "\[V26-FH\].* f6 " "logs_v26/smoke_${V}.log"
done
grep -l "Traceback" logs_v26/smoke_fh_*.log 2>/dev/null
echo "smoke finished"
