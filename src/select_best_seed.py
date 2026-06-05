"""
select_best_seed.py — ELBO-based unsupervised seed selection
=============================================================
Reads training logs for seeds 0-9.
Selects the seed with the highest (least negative) final ELBO.
No test labels used — pure unsupervised selection criterion.

Usage:
    python select_best_seed.py

Then re-evaluate best seed with both gamma_f=2.0 (default) and gamma_f=1.7:
    python eval_metrics.py --npz_path outputs/seeds/lda_..._seed{BEST}.npz \
        --gamma 2.0 --output_csv results/best_seed_gf2.0.csv ...
    python eval_metrics.py --npz_path outputs/seeds/lda_..._seed{BEST}.npz \
        --gamma 1.7 --output_csv results/best_seed_gf1.7.csv ...
"""
import os, glob
import pandas as pd
import numpy as np

def load_seed_results(results_dir='results/seeds', npz_dir='outputs/seeds'):
    records = []
    for seed in range(10):
        # Find eval CSV (try both gamma_f variants)
        csv = None
        for gf in ['2.0', '1.7', '']:
            suffix = f'_gf{gf}' if gf else ''
            path = f'{results_dir}/eval_seed{seed}{suffix}.csv'
            if os.path.exists(path):
                csv = path
                break
        if csv is None:
            continue

        df = pd.read_csv(csv)
        row = df.iloc[-1]
        records.append({
            'seed':       seed,
            'final_loss': row.get('final_loss', float('-inf')),
            'f1_macro':   row.get('f1_macro', 0.0),
            'nmi':        row.get('nmi', 0.0),
            'total_reward': row.get('total_reward', 0),
            'S':          row.get('S', 0),
            'csv':        csv,
        })
    return pd.DataFrame(records)

def main():
    df = load_seed_results()
    if df.empty:
        print('No seed results found in results/seeds/')
        return

    print('=== ALL SEEDS ===')
    print(f'{"Seed":>5}  {"Final Loss":>12}  {"F1":>8}  {"NMI":>8}  {"Rewards":>10}  {"S":>5}')
    print('-'*60)
    for _, r in df.sort_values('seed').iterrows():
        print(f'  {int(r.seed):>3}  {r.final_loss:>12.4f}  {r.f1_macro:>8.4f}  '
              f'{r.nmi:>8.4f}  {r.total_reward:>10.0f}  {int(r.S):>5}')

    # Select by ELBO (highest final_loss = least negative = best)
    best = df.loc[df['final_loss'].idxmax()]
    print()
    print(f'=== BEST SEED BY ELBO ===')
    print(f'  Seed:        {int(best.seed)}')
    print(f'  Final ELBO:  {best.final_loss:.4f}')
    print(f'  F1 (current gamma_f): {best.f1_macro:.4f}')
    print(f'  NMI:         {best.nmi:.4f}')
    print(f'  Total rewards: {best.total_reward:.0f}')

    # Find the npz for the best seed
    npz_pattern = f'outputs/seeds/lda_semeval14_rest*seed{int(best.seed)}*.npz'
    npzs = glob.glob(npz_pattern)
    if npzs:
        npz = sorted(npzs)[-1]
        print(f'  NPZ: {npz}')
        print()
        print('Re-evaluate this seed with both gamma_f values:')
        print()
        print(f'  python eval_metrics.py \\')
        print(f'      --npz_path {npz} \\')
        print(f'      --dataset semeval14_rest \\')
        print(f'      --data_path data/SemEval14/Restaurants_Test_Data_phaseB.xml \\')
        print(f'      --model_name_or_path bert-base-uncased \\')
        print(f'      --gamma 2.0 --importance_signal Rho --filter_mode hard \\')
        print(f'      --output_csv results/best_seed_gf2.0.csv')
        print()
        print(f'  python eval_metrics.py \\')
        print(f'      --npz_path {npz} \\')
        print(f'      --dataset semeval14_rest \\')
        print(f'      --data_path data/SemEval14/Restaurants_Test_Data_phaseB.xml \\')
        print(f'      --model_name_or_path bert-base-uncased \\')
        print(f'      --gamma 1.7 --importance_signal Rho --filter_mode hard \\')
        print(f'      --output_csv results/best_seed_gf1.7.csv')

    print()
    print('=== DISTRIBUTION SUMMARY ===')
    print(f'  N seeds evaluated: {len(df)}')
    print(f'  F1 mean +/- std:   {df.f1_macro.mean():.4f} +/- {df.f1_macro.std():.4f}')
    print(f'  F1 range:          {df.f1_macro.min():.4f} to {df.f1_macro.max():.4f}')
    print(f'  F1 best (by ELBO): {best.f1_macro:.4f}')
    print(f'  NMF baseline:       0.311')
    print(f'  Beats NMF (best):   {"YES" if best.f1_macro > 0.311 else "NO"}')

if __name__ == '__main__':
    main()
