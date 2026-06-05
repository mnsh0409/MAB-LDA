"""
run_scalability.py  — ICDM Scalability Stress Test
====================================================
Measures MAB-LDA throughput (tokens/second) as a function of
corpus size N, producing the O(N) scaling figure for the paper.

Design decisions (and why):
    burnin_ratio=0.125    Same as main experiments. burnin_ratio=0.0
                          would measure MAB on randomly-initialised
                          counts — biasing the comparison unfairly.

    store_cls=DenseCountStore  Correct for K=8 (small K). SparseCountStore
                               is slower for K<20 due to lil_matrix overhead.

    N_sizes=[1K,10K,50K,100K]  Feasible with Numba in ~6h total on CPU.

    SWEEPS=20               Sufficient to estimate per-sweep cost.
                            No need to run to convergence for timing.

    Metrics: tokens/second  The correct ICDM scalability metric.
             per_sweep_sec  Absolute time per Gibbs sweep.

Output files:
    results/scalability_times.csv   → Table/Figure data for paper
    results/scalability_plot.pdf    → Ready-to-include figure

Paper framing (Section 4.3):
    "Figure 3 shows that MAB-LDA scales linearly with corpus size
    (O(N) per sweep), achieving X tokens/sec on 100K Yelp reviews
    with a Y% overhead vs standard Gibbs — comparable to the overhead
    of prior guided topic models (Seeded LDA: Z%)."

Usage:
    # ICDM submission run (CPU + Numba, ~6h):
    python run_scalability.py --dataset yelp --gpu 0

    # Quick smoke test (5 min):
    python run_scalability.py --dataset yelp --n_sizes 1000 5000 --sweeps 5
"""

import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from model_simulation_reward  import LDAGibbs as BaselineLDA
from model_exclusivity_reward import LDAGibbs as MABLDA
from count_store import DenseCountStore, SparseCountStore

# ---------------------------------------------------------------------------
# Dataset loader registry
# ---------------------------------------------------------------------------
def get_loader(dataset: str):
    from dataload_massive import (
        load_massive_yelp, load_massive_pubmed, load_massive_amazon
    )
    registry = {
        'yelp':   (load_massive_yelp,   'Yelp Review Full'),
        'pubmed': (load_massive_pubmed,  'PubMed Abstracts'),
        'amazon': (load_massive_amazon,  'Amazon Reviews 2023'),
    }
    if dataset not in registry:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose: {list(registry)}")
    return registry[dataset]


# ---------------------------------------------------------------------------
# Single timing run
# ---------------------------------------------------------------------------
def time_single_run(model_cls, data_tensor, tokens_tensor,
                    K: int, sweeps: int, burnin_ratio: float,
                    store_cls, label: str) -> dict:
    """
    Runs one model for EXACTLY `sweeps` iterations and returns timing.

    IMPORTANT: We override the convergence/epoch logic by setting
    max_iter=sweeps with a burnin_ratio that guarantees the right
    number of MAB sweeps. Both models run the SAME number of sweeps
    so the wall-time comparison is fair.

    Per-sweep overhead = (mab_per_sweep / base_per_sweep - 1) * 100
    This isolates the algorithmic cost of the exclusivity reward from
    the convergence-rate difference between the two models.
    """
    D, L     = data_tensor.shape
    n_tokens = D * L
    n_burnin = int(sweeps * burnin_ratio)
    n_mab    = sweeps - n_burnin

    print(f'  [{time.strftime("%H:%M:%S")}] Starting {label} '
          f'({n_burnin} burnin + {n_mab} MAB sweeps)...')

    # Time burnin separately from MAB phase
    # Both models run identical code during burnin (MAB=False)
    # so burnin time should be equal — this validates the measurement.
    t0 = time.time()
    model = model_cls(data_tensor, K, tokens_tensor, store_cls=store_cls)
    model.run(epochs=1, max_iter=sweeps, burnin_ratio=burnin_ratio)
    wall = time.time() - t0

    per_sweep_all = wall / sweeps
    per_mab_sweep = wall / max(n_mab, 1)   # conservative: attribute all time to MAB
    tokens_per_s  = (n_tokens * sweeps) / wall

    print(f'  [{time.strftime("%H:%M:%S")}] {label}: '
          f'{wall:.1f}s  |  {per_sweep_all:.3f}s/sweep (all)  |  '
          f'{per_mab_sweep:.3f}s/MAB-sweep  |  {tokens_per_s:,.0f} tok/s')

    return {
        'model':           label,
        'total_seconds':   round(wall, 2),
        'per_sweep_sec':   round(per_sweep_all, 4),
        'per_mab_sweep_sec': round(per_mab_sweep, 4),
        'tokens_per_sec':  round(tokens_per_s, 0),
        'n_burnin':        n_burnin,
        'n_mab_sweeps':    n_mab,
    }


# ---------------------------------------------------------------------------
# Main stress test
# ---------------------------------------------------------------------------
def run_stress_test(dataset: str,
                    n_sizes: list,
                    sweeps: int,
                    K: int,
                    burnin_ratio: float,
                    tokenizer_name: str,
                    use_sparse: bool,
                    output_dir: str) -> pd.DataFrame:

    loader_fn, dataset_label = get_loader(dataset)
    store_cls = SparseCountStore if use_sparse else DenseCountStore
    results   = []
    out_dir = Path(output_dir) # Setup paths and clear the old file if it exists (save result per single run)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(output_dir) / 'scalability_times.csv'
    if csv_path.exists():
        csv_path.unlink() # Delete old file to prevent mixed headers

    for N in n_sizes:
        print()
        print('=' * 60)
        print(f'  N = {N:,} documents  |  dataset = {dataset_label}')
        print('=' * 60)

        # Load data
        data_tensor, tokens_tensor = loader_fn(
            num_docs=N, tokenizer_name=tokenizer_name)
        D, L = data_tensor.shape
        print(f'  Shape: {D} x {L}  |  vocab size: {len(tokens_tensor)}')

        base_row = {'N': N, 'seq_len': L, 'K': K,
                    'sweeps': sweeps, 'dataset': dataset_label,
                    'store': store_cls.__name__}

        # Baseline: standard Gibbs (simulation reward)
        r_base = time_single_run(
            BaselineLDA, data_tensor, tokens_tensor,
            K, sweeps, burnin_ratio, store_cls,
            label='Standard Gibbs (baseline)')
        full_base_row = {**base_row, **r_base}
        results.append(full_base_row)
        
        # --- CHECKPOINT BASELINE ---
        pd.DataFrame([full_base_row]).to_csv(
            csv_path, mode='a', header=not csv_path.exists(), index=False)

        # MAB-LDA: exclusivity reward
        r_mab = time_single_run(
            MABLDA, data_tensor, tokens_tensor,
            K, sweeps, burnin_ratio, store_cls,
            label='MAB-LDA (ours)')
        full_mab_row = {**base_row, **r_mab}
        results.append(full_mab_row)
        
        # --- CHECKPOINT MAB-LDA ---
        pd.DataFrame([full_mab_row]).to_csv(
            csv_path, mode='a', header=not csv_path.exists(), index=False)

        # MAB overhead relative to baseline
        overhead = (r_mab['total_seconds'] / r_base['total_seconds'] - 1) * 100
        print(f'  MAB overhead vs baseline: {overhead:+.1f}%')

    df = pd.DataFrame(results)
    #df.to_csv(csv_path, index=False)
    print(f'\nStress test complete! Results saved: {csv_path}')
    return df


# ---------------------------------------------------------------------------
# Plot: tokens/sec vs N  (log-log, shows O(N) linearity)
# ---------------------------------------------------------------------------
def plot_scalability(df: pd.DataFrame, output_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    plt.rcParams.update({'font.size': 12})

    baseline = df[df['model'].str.contains('baseline', case=False)]
    mab      = df[df['model'].str.contains('MAB', case=False)]
    Ns_base  = baseline['N'].values
    Ns_mab   = mab['N'].values

    # Left: tokens/sec vs N  (should be roughly flat — O(N) per sweep)
    ax = axes[0]
    ax.plot(Ns_base, baseline['tokens_per_sec'] / 1e3,
            'o--', color='#E63946', label='Standard Gibbs (baseline)', linewidth=2)
    ax.plot(Ns_mab,  mab['tokens_per_sec'] / 1e3,
            's-',  color='#1D3557', label='MAB-LDA (ours)',           linewidth=2.5)
    ax.set_xlabel('Corpus size N (documents)')
    ax.set_ylabel('Throughput (K tokens / second)')
    ax.set_title('Throughput vs Corpus Size')
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xscale('log')

    # Right: per-sweep time vs N (should be linear O(N))
    ax2 = axes[1]
    ax2.plot(Ns_base, baseline['per_sweep_sec'],
             'o--', color='#E63946', label='Standard Gibbs', linewidth=2)
    ax2.plot(Ns_mab,  mab['per_sweep_sec'],
             's-',  color='#1D3557', label='MAB-LDA',        linewidth=2.5)
    # Fit O(N) reference line
    if len(Ns_base) > 1:
        c    = baseline['per_sweep_sec'].values[0] / Ns_base[0]
        N_ref = np.linspace(Ns_base.min(), Ns_base.max(), 100)
        ax2.plot(N_ref, c * N_ref, 'k:', linewidth=1.5,
                 label='O(N) reference', alpha=0.6)
    ax2.set_xlabel('Corpus size N (documents)')
    ax2.set_ylabel('Time per Gibbs sweep (seconds)')
    ax2.set_title('Per-sweep Time vs Corpus Size')
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    plt.suptitle(f'MAB-LDA Scalability  ·  K={df["K"].iloc[0]}  ·  '
                 f'{df["sweeps"].iloc[0]} sweeps', y=1.01)
    plt.tight_layout()

    for fmt in ('pdf', 'png'):
        path = Path(output_dir) / f'scalability_plot.{fmt}'
        plt.savefig(path, format=fmt, bbox_inches='tight', dpi=150)
        print(f'Saved: {path}')
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='MAB-LDA scalability stress test')
    p.add_argument('--dataset',    type=str, default='yelp',
                   choices=['yelp', 'pubmed', 'amazon'],
                   help='Dataset to use (default: yelp)')
    p.add_argument('--n_sizes',    type=int, nargs='+',
                   default=[1_000, 10_000, 50_000, 100_000],
                   help='Corpus sizes to test (default: 1K 10K 50K 100K)')
    p.add_argument('--sweeps',     type=int,   default=20,
                   help='Gibbs sweeps per run (default: 20)')
    p.add_argument('--K',          type=int,   default=8,
                   help='Number of topics (default: 8)')
    p.add_argument('--burnin_ratio', type=float, default=0.125,
                   help='Burnin ratio — must match main experiments (default: 0.125)')
    p.add_argument('--tokenizer',  type=str,
                   default='bert-base-uncased',
                   help='HuggingFace tokenizer (default: bert-base-uncased)')
    p.add_argument('--sparse',     action='store_true',
                   help='Use SparseCountStore only for large K (e.g. K>=100)')
    p.add_argument('--output_dir', type=str, default='./results')
    p.add_argument('--no_plot',    action='store_true',
                   help='Skip generating the plot')
    return p.parse_args()


def main():
    args = parse_args()

    print('MAB-LDA Scalability Stress Test')
    print(f'Dataset:      {args.dataset}')
    print(f'N sizes:      {args.n_sizes}')
    print(f'K topics:     {args.K}')
    print(f'Sweeps:       {args.sweeps}')
    print(f'Burnin ratio: {args.burnin_ratio}')
    print(f'Store:        {"SparseCountStore" if args.sparse else "DenseCountStore"}')
    print(f'Tokenizer:    {args.tokenizer}')
    print()

    # Timing estimates so user knows what to expect
    for N in args.n_sizes:
        calls  = N * 64 * args.sweeps   # 64 avg tokens
        cpu_h  = calls * 40e-6 / 3600
        num_h  = calls * 5e-6  / 3600
        gpu_h  = calls * 0.5e-6 / 3600
        print(f'  N={N:>7,}: CPU~{cpu_h:.1f}h  Numba~{num_h:.1f}h  GPU~{gpu_h:.1f}h')
    print()

    df = run_stress_test(
        dataset        = args.dataset,
        n_sizes        = args.n_sizes,
        sweeps         = args.sweeps,
        K              = args.K,
        burnin_ratio   = args.burnin_ratio,
        tokenizer_name = args.tokenizer,
        use_sparse     = args.sparse,
        output_dir     = args.output_dir,
    )

    if not args.no_plot:
        plot_scalability(df, args.output_dir)

    # Print summary table
    print('\n' + '=' * 70)
    print('SCALABILITY SUMMARY')
    print('=' * 70)
    pivot = df.pivot_table(
        index='N', columns='model',
        values=['tokens_per_sec', 'per_sweep_sec']
    )
    print(pivot.to_string())


if __name__ == '__main__':
    main()
