# MAB-LDA: Multi-Armed Bandit Guided LDA for Unsupervised Aspect Discovery

> **ICDM 2026** — Research Track submission  
> SemEval-2014 Restaurant: F1 = 0.320 (ELBO-selected run, γ_f = 1.7)

---

## Overview

MAB-LDA augments collapsed Gibbs sampling with a Thompson Sampling bandit policy guided by a Herfindahl exclusivity reward. The reward silences subword fragments and stopwords automatically — without any preprocessing — enabling direct topic modelling on raw BERT subword vocabularies.

```
BERT subword tokens → Gibbs sampler → Herfindahl reward (HHI)
                                             ↓
                               ρ[w,k] EMA accumulation
                                             ↓
                            TD checkpoint (best ELBO sweep S)
                                             ↓
                            Inference: Rho[S] importance mask
                                             ↓
                              Hungarian-aligned F1 evaluation
```

---

## Results (SemEval-2014 Restaurant, K = 5)

| Method | Input | F1 macro |
|---|---|---|
| Standard LDA | word-level | 0.263 |
| NMF | word-level + stopwords | 0.311 |
| **MAB-LDA (ours)** | **raw BERT subwords** | **0.315** |
| Standard LDA (BERT) | raw BERT subwords | collapse |
| NMF (BERT) | raw BERT subwords | collapse |

ELBO-selected run across 10 random initialisations. No test labels used at any stage.

---

## Repository Structure

```
mab-lda/
├── model_base.py               # LDAGibbsBase — Gibbs sampler core
├── model_exclusivity_reward.py # LDAGibbs — Herfindahl reward + dynamic gate
├── dataload_SemEval.py         # SemEval XML → BERT token tensors
├── count_store.py              # CountStore backend (dense/sparse)
├── utils.py                    # Shared utilities
├── utils_logging.py            # Sweep-level CSV logger
├── run_all.py                  # Train MAB-LDA on one dataset
├── eval_metrics.py             # Evaluate saved .npz with Hungarian alignment
├── run_seeds_sweep.sh          # 5-seed × 5-dataset standardisation sweep
├── select_best_seed.py         # ELBO-based seed selection
├── run_subword_baselines.py    # LDA(BERT) and NMF(BERT) collapse demo
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Verify installation
pytest tests/ -v         # 24 tests, no real data required

git clone https://github.com/<your-username>/mab-lda.git
cd mab-lda
pip install -r requirements.txt
```

Python 3.10–3.12. No GPU required — all experiments run on CPU.

---

## Data

Download the SemEval ABSA shared-task XML files and place them as follows:

```
data/
├── SemEval14/
│   ├── Restaurants_Train_v2.xml
│   ├── Restaurants_Test_Data_phaseB.xml
│   ├── Laptop_Train_v2.xml
│   └── Laptop_Test_Data_phaseB.xml
├── SemEval15/
│   ├── ABSA15_Restaurants_Train_Final.xml
│   └── ABSA15_Restaurants_Test.xml
├── SemEval16/
│   ├── ABSA16_Restaurants_Train_SB1_v2.xml
│   └── EN_REST_SB1_TEST.xml.gold
└── MAMS/
    ├── train.xml
    └── test.xml
```

SemEval data is available from the [SemEval shared task organisers](http://alt.qcri.org/semeval2016/task5/).  
MAMS is available from [MAMS GitHub](https://github.com/siat-nlp/MAMS-for-ABSA).

---

## Quickstart: Reproduce the Main Result

### Single run (Rest-14, seed 0)

```bash
export MAB_SEED=0

python run_all.py \
    --dataset    semeval14_rest \
    --train_path data/SemEval14/Restaurants_Train_v2.xml \
    --test_path  data/SemEval14/Restaurants_Test_Data_phaseB.xml \
    --model_name_or_path bert-base-uncased \
    --gamma 2.0 --epochs 10 --iters 400 \
    --output_dir outputs/

python eval_metrics.py \
    --npz_path   outputs/lda_semeval14_rest_5k_400iter_seed0.npz \
    --dataset    semeval14_rest \
    --data_path  data/SemEval14/Restaurants_Test_Data_phaseB.xml \
    --model_name_or_path bert-base-uncased \
    --gamma 1.7 --importance_signal Rho --filter_mode hard \
    --output_csv results/rest14_seed0.csv
```

### 5-seed × 5-dataset standardisation (overnight, ~12 hours)

```bash
bash run_seeds_sweep.sh 2>&1 | tee results/sweep_log.txt
```

Results are written to `results/5_seed_standardization.csv`.

### Select best seed by ELBO

```bash
python select_best_seed.py
```

### Reproduce scalability results (Table VIII)

```python
from dataload_massive import load_massive_yelp

data, tokens = load_massive_yelp(num_docs=10_000)
# Then pass data and tokens to LDAGibbs as usual
```

Requires `pip install datasets` for HuggingFace streaming.

### Reproduce Table X (aviation domain transfer)

```bash
# Step 1: Train (produces outputs/lda_results_aviation.npz)
python run_domain.py aviation \
    --tokenizer bert-base-uncased \
    --n_topics 8 --epochs 5 --iters 800

# Step 2: Extract topic words (Table X)
python extract_domain_topics.py \
    --mab_npz  outputs/lda_results_aviation.npz \
    --base_npz outputs/lda_results_aviation_baseline.npz \
    --top_n 8
```

The aviation airline reviews dataset is publicly available on IEEE Data Port:
search **"Top 10 Airlines Reviews 2016-2023"** and place the file at
`data/Top10_airlines_reviews(2016-2023Jul).txt`. The script works on any corpus trained with `run_all.py` — replace `--mab_npz` with your own `.npz` path.

### Subword collapse demonstration

```bash
python run_subword_baselines.py \
    --train_path data/SemEval14/Restaurants_Train_v2.xml \
    --test_path  data/SemEval14/Restaurants_Test_Data_phaseB.xml \
    --model_name_or_path bert-base-uncased
```

---

## Key Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `--gamma` | 2.0 | γ_train: Herfindahl reward threshold during training |
| `--gamma` (eval) | 1.7 | γ_f: inference filter threshold (sensitivity analysis) |
| `--epochs` | 10 | Number of training epochs |
| `--iters` | 400 | Gibbs sweeps per epoch |
| `--burnin_ratio` | 0.125 | Fraction of sweeps before MAB activates |
| `MAB_SEED` (env) | 0–9 | Random seed (set via environment variable) |

---

## Model Selection Protocol

Due to the stochasticity of collapsed Gibbs sampling and MAB exploration dynamics, individual training trajectories converge to different local optima. Following standard practice in non-convex optimisation and reinforcement learning, we run multiple trajectory searches and report the run achieving the minimum variational loss (highest ELBO). No test labels are used at any selection stage.

The temporal difference (TD) checkpoint in `model_base.py` implements the same principle within a single run — selecting the best sweep `S` by training ELBO.

---

## Citation

```bibtex
@inproceedings{mab_lda_icdm2026,
  title     = {MAB-LDA: Multi-Armed Bandit Guided Latent Dirichlet Allocation
               for Unsupervised Aspect Discovery},
  booktitle = {Proceedings of the IEEE International Conference on Data Mining
               (ICDM)},
  year      = {2026},
}
```

---

## License

MIT License. See [LICENSE](LICENSE).

The SemEval and MAMS datasets are subject to their own respective licenses and are not redistributed here.
