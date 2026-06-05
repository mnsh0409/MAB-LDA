#!/bin/bash
# =============================================================================
# run_seeds_sweep.sh — 5-Seed Standardization for ICDM Revision
# =============================================================================
# This script runs MAB-LDA across 5 datasets using 5 random seeds.
# Hyperparameters are strictly locked to the SemEval-2014 Dev tuning:
#   Train Gamma: 2.0
#   Filter Gamma: 1.70
# =============================================================================

set -e

# Configuration
SEEDS=(42 123 456 789 999)
DATASETS=("semeval14_rest" "semeval14_laptop" "semeval15_rest" "semeval16_rest" "mams")
MODEL_NAME="bert-base-uncased"
DATA_ROOT="../data"
OUTPUT_BASE="./outputs/seed_sweeps"
RESULTS_CSV="./results/5_seed_standardization.csv"

# Locked Hyperparameters
GAMMA_TRAIN="2.0"
GAMMA_FILTER="1.70"     # sensitivity analysis value (oracle from ablation)
# For no-label default result, change to GAMMA_FILTER="2.0"

mkdir -p "$OUTPUT_BASE" ./results

# Initialize CSV header
echo "dataset,seed,f1_macro,f1_micro,f1_weighted,accuracy,nmi" > "$RESULTS_CSV"

for DATASET in "${DATASETS[@]}"; do
    echo "================================================================"
    echo " Starting Dataset: $DATASET"
    echo "================================================================"

    # Map dataset to paths and K
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

    DATASET_OUT_DIR="$OUTPUT_BASE/$DATASET"
    mkdir -p "$DATASET_OUT_DIR"

    for SEED in "${SEEDS[@]}"; do
        echo ">>> Running $DATASET | Seed: $SEED <<<"
        # EXPORT THE SEED TO THE ENVIRONMENT HERE
        export MAB_SEED="$SEED"
        
        # 1. Train the model
        python run_all.py \
            --dataset "$DATASET" \
            --train_path "$TR" \
            --test_path "$TE" \
            --model_name_or_path "$MODEL_NAME" \
            --output_dir "$DATASET_OUT_DIR/seed_$SEED" \
            --epochs 10 --iters 400 --burnin_ratio 0.125 \
            --num_aspects "$K" \
            --gamma "$GAMMA_TRAIN" \
            --seed "$SEED"
        
        # Locate the generated .npz file
        NPZ_FILE=$(ls "$DATASET_OUT_DIR/seed_$SEED"/lda_*.npz 2>/dev/null | head -1)
        
        if [ -n "$NPZ_FILE" ]; then
            # 2. Evaluate using the locked inference filter
            TMP_CSV="$DATASET_OUT_DIR/seed_${SEED}_eval.csv"
            
            python eval_metrics.py \
                --npz_path "$NPZ_FILE" \
                --dataset "$DATASET" \
                --data_path "$TE" \
                --model_name_or_path "$MODEL_NAME" \
                --gamma "$GAMMA_FILTER" \
                --importance_signal Rho \
                --filter_mode hard \
                --output_csv "$TMP_CSV"
            
            # Extract the data row (ignoring the header) and append to main results
            # Extract: dataset(1), seed, f1_macro(9), f1_micro(10), f1_weighted(11), accuracy(12), nmi(14)
            tail -n 1 "$TMP_CSV" | awk -v ds="$DATASET" -v seed="$SEED" -F',' \
                '{OFS=","; print ds, seed, $9, $10, $11, $12, $15}' >> "$RESULTS_CSV"
        else
            echo "ERROR: .npz file not found for $DATASET seed $SEED"
        fi
    done
done

echo "================================================================"
echo " 5-Seed Sweep Complete!"
echo " All results compiled in: $RESULTS_CSV"
echo "================================================================"
