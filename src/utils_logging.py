"""
utils_logging.py
================
Sweep-level experiment logger for MAB-LDA.

Replaces the timer-based background logger with a direct callback
called once per sweep from inside run(). Records everything needed
for the paper tables and convergence figures.

Usage:
    from utils_logging import SweepLogger

    logger = SweepLogger(
        log_path   = './logs/semeval14_rest_run1.csv',
        dataset    = 'semeval14_rest',
        model      = 'MAB-LDA',
        n_docs     = 3000,
        n_topics   = 5,
    )

    # In run() after each sweep:
    logger.log_sweep(
        sweep          = s,
        epoch          = epoch,
        mab_active     = MAB,
        loss           = self.loss,
        break_count    = break_,
        sweep_regret   = total_sweep_regret,
        epoch_regret   = total_epoch_regret,
        tokens_per_sec = tokens_per_sec,
        alpha_entropy  = float(np.sum(np.square(self.alpha / self.alpha.sum()))),
        beta_entropy   = float(np.sum(np.square(self.beta  / self.beta.sum()))),
    )

    logger.close()

Output CSV columns (one row per sweep):
    timestamp, dataset, model, n_docs, K,
    sweep, epoch, mab_active,
    loss, loss_delta,
    break_count, sweep_regret, epoch_regret,
    tokens_per_sec, alpha_entropy, beta_entropy,
    wall_sec_since_start
"""

import csv
import time
import os
from pathlib import Path


class SweepLogger:
    """
    Logs one row per Gibbs sweep to a CSV file.
    No background threads. No timers. Fires only when called.
    """

    FIELDS = [
        'timestamp', 'dataset', 'model', 'n_docs', 'K',
        'sweep', 'epoch', 'mab_active',
        'loss', 'loss_delta',
        'break_count', 'sweep_regret', 'epoch_regret',
        'tokens_per_sec', 'alpha_entropy', 'beta_entropy',
        'wall_sec_since_start',
    ]

    def __init__(self, log_path: str, dataset: str, model: str,
                 n_docs: int, n_topics: int):
        self.log_path   = Path(log_path)
        self.dataset    = dataset
        self.model      = model
        self.n_docs     = n_docs
        self.n_topics   = n_topics
        self._start_t   = time.time()
        self._prev_loss = None

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._f   = open(self.log_path, 'w', newline='')
        self._csv = csv.DictWriter(self._f, fieldnames=self.FIELDS)
        self._csv.writeheader()
        self._f.flush()
        print(f'[SweepLogger] Logging to {self.log_path}')

    def log_sweep(self,
                  sweep:          int,
                  epoch:          int,
                  mab_active:     bool,
                  loss:           float,
                  break_count:    int   = 0,
                  sweep_regret:   float = 0.0,
                  epoch_regret:   float = 0.0,
                  tokens_per_sec: float = 0.0,
                  alpha_entropy:  float = 0.0,
                  beta_entropy:   float = 0.0) -> None:

        loss_delta = (loss - self._prev_loss) if self._prev_loss is not None else 0.0
        self._prev_loss = loss

        row = {
            'timestamp':           time.strftime('%Y-%m-%d %H:%M:%S'),
            'dataset':             self.dataset,
            'model':               self.model,
            'n_docs':              self.n_docs,
            'K':                   self.n_topics,
            'sweep':               sweep,
            'epoch':               epoch,
            'mab_active':          int(mab_active),
            'loss':                round(loss, 6),
            'loss_delta':          round(loss_delta, 6),
            'break_count':         break_count,
            'sweep_regret':        round(sweep_regret, 4),
            'epoch_regret':        round(epoch_regret, 4),
            'tokens_per_sec':      round(tokens_per_sec, 0),
            'alpha_entropy':       round(alpha_entropy, 6),
            'beta_entropy':        round(beta_entropy, 6),
            'wall_sec_since_start': round(time.time() - self._start_t, 1),
        }
        self._csv.writerow(row)
        self._f.flush()   # write immediately so log survives crashes

        # Print one-line summary to stdout
        mab_str = 'MAB' if mab_active else 'BRN'
        print(f'[{row["timestamp"]}] '
              f'sweep={sweep:3d}  ep={epoch}  {mab_str}  '
              f'loss={loss:+.4f} (Δ{loss_delta:+.4f})  '
              f'break={break_count:4d}  '
              f'regret={sweep_regret:.3f}  '
              f'tok/s={tokens_per_sec:,.0f}')

    def close(self):
        self._f.close()
        print(f'[SweepLogger] Closed. {self.log_path}')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Summary: read a completed log and print paper-table numbers
# ---------------------------------------------------------------------------
def summarise_log(log_path: str) -> None:
    """Print key numbers from a completed sweep log for paper writing."""
    import pandas as pd

    df = pd.read_csv(log_path)
    if df.empty:
        print('Log is empty.')
        return

    print(f'\n{"="*60}')
    print(f'Log summary: {log_path}')
    print(f'{"="*60}')
    print(f'Total sweeps logged:  {len(df)}')
    print(f'Dataset:              {df["dataset"].iloc[0]}')
    print(f'Model:                {df["model"].iloc[0]}')
    print(f'N docs:               {df["n_docs"].iloc[0]:,}')
    print(f'K topics:             {df["K"].iloc[0]}')
    print()

    real = df[df['loss'] < 0]
    if not real.empty:
        print(f'Loss trajectory:')
        print(f'  Initial loss:     {real["loss"].iloc[0]:+.4f}')
        print(f'  Final loss:       {real["loss"].iloc[-1]:+.4f}')
        print(f'  Best loss:        {real["loss"].max():+.4f}')
        print(f'  Total improvement:{real["loss"].iloc[-1] - real["loss"].iloc[0]:+.4f}')
        print()

    mab = df[df['mab_active'] == 1]
    if not mab.empty:
        print(f'MAB phase ({len(mab)} sweeps):')
        print(f'  Total regret:     {mab["sweep_regret"].sum():.2f}')
        print(f'  Mean regret/sweep:{mab["sweep_regret"].mean():.4f}')
        print(f'  break_ max:       {mab["break_count"].max()}')
        print()

    if 'tokens_per_sec' in df.columns:
        tps = df[df['tokens_per_sec'] > 0]['tokens_per_sec']
        if not tps.empty:
            print(f'Throughput:')
            print(f'  Mean tok/s:       {tps.mean():,.0f}')
            print(f'  Min tok/s:        {tps.min():,.0f}')
            print()

    total_wall = df['wall_sec_since_start'].max()
    print(f'Total wall time:      {total_wall/3600:.2f} hours ({total_wall:.0f}s)')


# ---------------------------------------------------------------------------
# Stub kept for backward compatibility with old run_scalability.py import
# ---------------------------------------------------------------------------
def setup_experiment_logger(*args, **kwargs):
    """Legacy stub — use SweepLogger instead."""
    pass

def log_system_metrics(*args, **kwargs):
    """Legacy stub — use SweepLogger.log_sweep() instead."""
    pass
