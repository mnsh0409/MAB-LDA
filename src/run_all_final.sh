#!/bin/bash
# =============================================================================
# run_all_final.sh  — ICDM complete submission sweep
# =============================================================================
# Runs EVERYTHING needed for the ICDM paper in the correct order:
#
#   Step 1:  Main results  (Table 1) — MAB-LDA on all 6 datasets
#   Step 2:  Ablations     (Table 3) — reward, gate, gamma, burnin, backbone
#   Step 3:  Baselines     (Table 1) — standard LDA, NMF, BTM, Seeded LDA
#   Step 4:  Evaluation    (Table 1,2,3) — F1/purity/NMI/NPMI from .npz files
#   Step 5:  Merge tables  — combine MAB-LDA + baseline results into one CSV
#
# Total estimated time with Numba + bert-base-uncased on a GPU machine:
#   Step 1: ~6 hours   Step 2: ~8 hours   Step 3: ~2 hours   Step 4: ~20 min
#
# Edit MODEL_NAME_DEV (fast, for iteration) and MODEL_NAME_FINAL (paper runs).
# Run with MODEL_NAME=$MODEL_NAME_DEV first to check everything works,
# then switch to MODEL_NAME=$MODEL_NAME_FINAL for submission numbers.
# =============================================================================

set -e   # stop on first failure — do not silently continue past errors

# --------------- EDIT THESE ---------------------------------------------------
DATA_ROOT="../data"

MODEL_NAME_DEV="bert-base-uncased"          # fast iteration runs (~6h total)
MODEL_NAME_FINAL="bert-base-uncased"         # set to LOCAL_LLAMA for LLaMA runs

MODEL_NAME="$MODEL_NAME_DEV"                # <- change to MODEL_NAME_FINAL for submission

# Local LLaMA paths — set to your snapshot directories
# Find snapshots: ls ~/.cache/huggingface/hub/models--meta-llama--*/snapshots/
# Auto-detect LLaMA snapshot paths — gracefully skip if not available
# Auto-detect HuggingFace cache location
# Linux/Mac: ~/.cache/huggingface/hub
# WSL (Windows Subsystem for Linux): /mnt/c/Users/<WindowsUser>/.cache
_HF_ROOT="$HOME/.cache/huggingface/hub"
if [ -d "/mnt/c/Users" ]; then
    # WSL detected — $USER is your WSL username (may differ from Windows username)
    # If models not found, replace $USER with your Windows username manually:
    #   _HF_ROOT="/mnt/c/Users/YourWindowsName/.cache/huggingface/hub"
    _WIN_USER=$(cmd.exe /c "echo %USERNAME%" 2>/dev/null | tr -d "\r\n")
    _WIN_USER=${_WIN_USER:-$USER}   # fallback to $USER if cmd.exe fails
    _HF_ROOT="/mnt/c/Users/$_WIN_USER/.cache/huggingface/hub"
fi

# Find snapshots: ls ~/.cache/huggingface/hub/models--meta-llama--*/snapshots/
echo "HuggingFace cache: $_HF_ROOT"
_SNAP_8B=$(ls ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/ 2>/dev/null | head -1)
_SNAP_1B=$(ls ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/    2>/dev/null | head -1)
_SNAP_3B=$(ls ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B/snapshots/    2>/dev/null | head -1)

LOCAL_LLAMA="$HOME/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/$_SNAP_8B"
LOCAL_LLAMA_1B="$HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/$_SNAP_1B"
LOCAL_LLAMA_3B="$HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B/snapshots/$_SNAP_3B"

EPOCHS=10
ITERS=400
BURNIN_RATIO=0.125
GAMMA=2.0
OUTPUT_BASE="./outputs"
RESULTS_DIR="./results"
# ------------------------------------------------------------------------------

mkdir -p "$OUTPUT_BASE" "$RESULTS_DIR" ./logs

header() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
    date
}


# ==============================================================================
# STEP 2 — ABLATIONS (Table 3)
# ==============================================================================
header "STEP 2: Ablations"
ABLATION_BASE="$OUTPUT_BASE/ablation"
TRAIN14="$DATA_ROOT/SemEval14/Restaurants_Train_v2.xml"
TEST14="$DATA_ROOT/SemEval14/Restaurants_Test_Data_phaseB.xml"


echo "Done: backbone sweep"

# F: NPMI vs sweeps (for 20NG — classical shadow validation)
for N_ITERS in 50 100 200 400; do
    echo "Ablation F: NPMI at $N_ITERS iters"
    python run_all.py \
        --dataset ng20 --model_name_or_path "$MODEL_NAME" \
        --output_dir "$ABLATION_BASE/npmi_${N_ITERS}" \
        --epochs 5 --iters "$N_ITERS" --burnin_ratio 0.2 \
        --num_aspects 20 --gamma $GAMMA --ng20_max_docs 2000
done
echo "Done: NPMI sweep"

header "STEP 2 complete"

# ==============================================================================
# STEP 3 — BASELINES (Table 1: comparison methods)
# ==============================================================================
header "STEP 3: Baseline comparisons"

for DATASET in semeval14_rest semeval14_laptop semeval15_rest semeval16_rest mams; do
    case $DATASET in
        semeval14_rest)
            TR="$DATA_ROOT/SemEval14/Restaurants_Train_v2.xml"
            TE="$DATA_ROOT/SemEval14/Restaurants_Test_Data_phaseB.xml"
            K=5 ;;
        semeval14_laptop)
            TR="$DATA_ROOT/SemEval14/Laptop_Train_v2.xml"
            TE="$DATA_ROOT/SemEval14/Laptop_Test_Data_phaseB.xml"
            K=5 ;;
        semeval15_rest)
            TR="$DATA_ROOT/SemEval15/ABSA-15_Restaurants_Train_Final.xml"
            TE="$DATA_ROOT/SemEval15/ABSA15_Restaurants_Test.xml"
            K=5 ;;
        semeval16_rest)
            TR="$DATA_ROOT/SemEval16/ABSA16_Restaurants_Train_SB1_v2.xml"
            TE="$DATA_ROOT/SemEval16/EN_REST_SB1_TEST.xml.gold"
            K=5 ;;
        mams)
            TR="$DATA_ROOT/MAMS/train.xml"
            TE="$DATA_ROOT/MAMS/test.xml"
            K=8 ;;
    esac

    echo "Baselines for $DATASET (K=$K)..."
    python run_baselines.py \
        --dataset "$DATASET" \
        --train_path "$TR" \
        --test_path  "$TE" \
        --model_name_or_path "$MODEL_NAME" \
        --num_aspects "$K" \
        --output_csv "$RESULTS_DIR/baselines_all.csv" \
        --skip_btm   # BTM is slow on >3K docs; remove --skip_btm for full run
    echo "  Done: $DATASET baselines"
done

header "STEP 3b: ABAE baselines"
for DATASET in semeval14_rest semeval14_laptop semeval15_rest semeval16_rest mams; do
    case $DATASET in
        semeval14_rest)   TR="$TRAIN14"; TE="$TEST14"; K=5 ;;
        semeval14_laptop) TR="$DATA_ROOT/SemEval14/Laptop_Train_v2.xml"
                          TE="$DATA_ROOT/SemEval14/Laptop_Test_Data_phaseB.xml"; K=5 ;;
        semeval15_rest)   TR="$DATA_ROOT/SemEval15/ABSA-15_Restaurants_Train_Final.xml"
                          TE="$DATA_ROOT/SemEval15/ABSA15_Restaurants_Test.xml"; K=5 ;;
        semeval16_rest)   TR="$DATA_ROOT/SemEval16/ABSA16_Restaurants_Train_SB1_v2.xml"
                          TE="$DATA_ROOT/SemEval16/EN_REST_SB1_TEST.xml.gold"; K=5 ;;
        mams)             TR="$DATA_ROOT/MAMS/train.xml"
                          TE="$DATA_ROOT/MAMS/test.xml"; K=8 ;;
    esac
    python run_abae.py \
        --dataset "$DATASET" --train_path "$TR" --test_path "$TE" \
        --model_name_or_path "$MODEL_NAME" \
        --num_aspects "$K" --epochs 200 --patience 10 \
        --output_csv "$RESULTS_DIR/abae_results.csv"
done

header "STEP 3 complete"

# ==============================================================================
# STEP 4 — EVALUATION (generate all metric rows)
# ==============================================================================
header "STEP 4: Evaluate all MAB-LDA outputs"

# Main results
for DATASET in semeval14_rest semeval14_laptop semeval15_rest semeval16_rest mams; do
    case $DATASET in
        semeval14_rest)   DP="$DATA_ROOT/SemEval14/Restaurants_Test_Data_phaseB.xml" ; K=5 ;;
        semeval14_laptop) DP="$DATA_ROOT/SemEval14/Laptop_Test_Data_phaseB.xml" ; K=5 ;;
        semeval15_rest)   DP="$DATA_ROOT/SemEval15/ABSA15_Restaurants_Test.xml" ; K=5 ;;
        semeval16_rest)   DP="$DATA_ROOT/SemEval16/EN_REST_SB1_TEST.xml.gold" ; K=5 ;;
        mams)             DP="$DATA_ROOT/MAMS/test.xml" ; K=8 ;;
    esac
    NPZ=$(ls "$OUTPUT_BASE/$DATASET"/lda_*${GAMMA}.npz 2>/dev/null | head -1)
    if [ -n "$NPZ" ]; then
        python eval_metrics.py \
            --npz_path "$NPZ" --dataset "$DATASET" \
            --data_path "$DP" \
            --model_name_or_path "$MODEL_NAME" \
            --gamma         2.05 \
            --importance_signal Rho \
            --filter_mode       hard \
            --output_csv "$RESULTS_DIR/mabslda_results.csv"
    fi
done

# 20NG with NPMI
NPZ20=$(ls "$OUTPUT_BASE/ng20"/lda_*${GAMMA}.npz 2>/dev/null | head -1)
if [ -n "$NPZ20" ]; then
    python eval_metrics.py \
        --npz_path "$NPZ20" --dataset ng20 \
        --model_name_or_path "$MODEL_NAME" \
        --compute_npmi \
        --output_csv "$RESULTS_DIR/mabslda_results.csv"
fi

# NPMI sweep results
for N_ITERS in 50 100 200 400; do
    NPZ=$(ls "$ABLATION_BASE/npmi_${N_ITERS}"/lda_*${GAMMA}.npz 2>/dev/null | head -1)
    if [ -n "$NPZ" ]; then
        python eval_metrics.py \
            --npz_path "$NPZ" --dataset ng20 \
            --model_name_or_path "$MODEL_NAME" \
            --compute_npmi \
            --gamma         2.05 \
            --importance_signal Rho \
            --filter_mode       hard \
            --output_csv "$RESULTS_DIR/npmi_vs_iters.csv"
    fi
done

# Backbone ablation evaluation (bert / roberta / electra)
# Uses semeval14_rest test split — same dataset as main results
# Each backbone uses its OWN model_name_or_path for tokenizer consistency
DP_REST14="$DATA_ROOT/SemEval14/Restaurants_Test_Data_phaseB.xml"

declare -A BACKBONE_MODELS=(
    [bert-base-uncased]="bert-base-uncased"
    [roberta-base]="roberta-base"
    [google_electra-base-discriminator]="google/electra-base-discriminator"
    [microsoft_deberta-v3-base]="microsoft/deberta-v3-base"
    [llama_8b]="$LOCAL_LLAMA"
    [llama_1b]="$LOCAL_LLAMA_1B"
    [llama_3b]="$LOCAL_LLAMA_3B"
)

for SAFE_BB in bert-base-uncased roberta-base google_electra-base-discriminator \
              microsoft_deberta-v3-base \
              llama_8b llama_1b llama_3b; do
    BB_MODEL="${BACKBONE_MODELS[$SAFE_BB]}"
    NPZ=$(ls "$ABLATION_BASE/backbone_${SAFE_BB}"/lda_*${GAMMA}.npz 2>/dev/null | head -1)
    if [ -n "$NPZ" ]; then
        echo "Evaluating backbone: $BB_MODEL"
        python eval_metrics.py \
            --npz_path "$NPZ" \
            --dataset semeval14_rest \
            --data_path "$DP_REST14" \
            --model_name_or_path "$BB_MODEL" \
            --gamma         2.05 \
            --importance_signal Rho \
            --filter_mode       hard \
            --output_csv "$RESULTS_DIR/backbone_ablation.csv"
    else
        echo "Backbone npz not found, skipping: $SAFE_BB"
    fi
done

header "STEP 4 complete"

# ==============================================================================
# STEP 5 — MERGE AND PRINT PAPER TABLES
# ==============================================================================
header "STEP 5: Generate paper tables"

python - <<'PYEOF'
import pandas as pd
from pathlib import Path

results_dir = Path('./results')

# Table 1: MAB-LDA vs all baselines
mab = pd.read_csv(results_dir / 'mabslda_results.csv') if (results_dir/'mabslda_results.csv').exists() else pd.DataFrame()
bl  = pd.read_csv(results_dir / 'baselines_all.csv')  if (results_dir/'baselines_all.csv').exists()  else pd.DataFrame()

if not mab.empty:
    mab['method'] = 'MAB-LDA (Ours)'

table1 = pd.concat([bl, mab], ignore_index=True)
if not table1.empty:
    cols = ['dataset','method','f1_macro','f1_micro','f1_weighted','accuracy','nmi']
    cols = [c for c in cols if c in table1.columns]
    print('\n' + '='*70)
    print('TABLE 1: Main comparison results')
    print('='*70)
    print(table1[cols].sort_values(['dataset','method']).to_string(index=False))
    table1[cols].to_csv(results_dir / 'table1_main_results.csv', index=False)
    print(f'\nSaved: results/table1_main_results.csv')

# NPMI vs iters (for paper Figure 2)
npmi = pd.read_csv(results_dir / 'npmi_vs_iters.csv') if (results_dir/'npmi_vs_iters.csv').exists() else pd.DataFrame()
if not npmi.empty:
    print('\n' + '='*70)
    print('FIGURE 2: NPMI coherence vs number of sweeps (20 Newsgroups, K=20)')
    print('='*70)
    print(npmi[['dataset','npmi']].to_string(index=False) if 'npmi' in npmi.columns else 'NPMI column not found')
PYEOF

echo ""
echo "================================================================"
echo "ALL STEPS COMPLETE"
echo "================================================================"
echo "Paper tables in: $RESULTS_DIR/"
echo "  table1_main_results.csv  → Table 1 (paste into paper)"
echo "  baselines_all.csv        → baseline rows"
echo "  mabslda_results.csv      → MAB-LDA rows"
echo "  backbone_ablation.csv    → backbone rows"
echo "  npmi_vs_iters.csv        → Figure 2 data"
echo ""
echo "Next: write the paper. Start with Section 4 (experiments)"
echo "using the numbers in table1_main_results.csv."
date
