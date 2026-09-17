# Step-by-Step Reproduction Guide

## STEP 0 - Hardware requirements

    CPU: 8+ cores recommended
    RAM: 32+ GiB
    GPU: 1x NVIDIA GPU with 24+ GiB VRAM
    OS:  Ubuntu 24.04 LTS
    Storage: 50+ GiB free

## STEP 1 - Clone and create virtual environment

    git clone https://github.com/klujac/cbm-p1ts-review
    cd cbm-p1ts-review
    python3 -m venv .venv
    source .venv/bin/activate
    export PYTHONNOUSERSITE=1
    pip install --no-cache-dir -r env/requirements_frozen_v25_1.txt

## STEP 2 - Download TabPFN-3 checkpoint

    mkdir -p tabpfn_models
    python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Prior-Labs/TabPFN', filename='tabpfn-v3-classifier-v3_default.ckpt', local_dir='tabpfn_models')"

Verify SHA-256:

    python3 -c "import hashlib; h=hashlib.sha256(); [h.update(c) for c in iter(open('tabpfn_models/tabpfn-v3-classifier-v3_default.ckpt','rb').read1, b'')]; print('MATCH' if h.hexdigest()=='d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988' else 'MISMATCH')"

Set environment variables:

    export LDRV25_TABPFN_MODEL_PATH=$(pwd)/tabpfn_models/tabpfn-v3-classifier-v3_default.ckpt
    export LDRV25_REQUIRE_TABPFN_TOKEN=0

## STEP 3 - Smoke test (mandatory)

    python3 src/xgboost_spawn_smoke_test.py
    # Expected: XGBOOST_SPAWN_TEST_OK

## STEP 4 - Run full benchmark (approx. 5 days on 2x H200)

    tmux new -s benchmark
    source .venv/bin/activate
    export PYTHONNOUSERSITE=1
    export LDRV25_TABPFN_MODEL_PATH=$(pwd)/tabpfn_models/tabpfn-v3-classifier-v3_default.ckpt
    export LDRV25_REQUIRE_TABPFN_TOKEN=0
    python -u src/benchmark_driver_v25_3.py --no-menu --datasets-group 2 --trials 40 --epochs-hpo 80 --patience 40 --epochs-final 250 --restarts 2 --seeds 42,43,44 2>&1 | tee benchmark_run.log

Detach: Ctrl+B then D
Resume after interruption: same command (auto-resumes from checkpoint)
Protocol hash must remain: 9e2fd6f6c51d2e9150021e0e2ef5e8c539775437f6c81a0950760b7b4987819e

## STEP 5 - Verify results

    python3 -c "import pandas as pd; df=pd.read_csv('results/comparison.csv'); print('Rows:',len(df),'(expected 9600)'); print('OK:',(df.Status=='OK').sum(),'(expected 9360)'); ds=df.groupby(['Algorithm','Dataset'])['AUC_ROC'].mean(); print('CBM-KE AUC:',round(ds['CBM_KE'].mean(),3),'(original submission: 0.935; revised value in REPRODUCE_v26.md)')"

## STEP 6 - Generate diabetes rules (approx. 2 hours)

    tmux new -s rules
    source .venv/bin/activate
    export LDRV25_TABPFN_MODEL_PATH=$(pwd)/tabpfn_models/tabpfn-v3-classifier-v3_default.ckpt
    export LDRV25_REQUIRE_TABPFN_TOKEN=0
    python -u src/rule_gen_v25_1.py --mode normal 2>&1 | tee rule_gen.log

NOTE: rule_gen_v25_1.py already contains patch CBM_PATIENCE_ES=40 on line 385.

## STEP 7 - Compile the paper

    cd paper/
    pdflatex article_v25_IS.tex && bibtex article_v25_IS && pdflatex article_v25_IS.tex && pdflatex article_v25_IS.tex

Requires: sudo apt install texlive-full

## Troubleshooting

    SIGSEGV in XGBoost         -> run smoke test first; use Python 3.12+
    NameError: CBM_PATIENCE_ES -> use rule_gen_v25_1.py from this repo (line 385)
    TABPFN_TOKEN is not set    -> export LDRV25_REQUIRE_TABPFN_TOKEN=0
    SHA-256 mismatch           -> re-download checkpoint from HuggingFace
    Workers CRASHED rc=0       -> normal clean exit; check --status
