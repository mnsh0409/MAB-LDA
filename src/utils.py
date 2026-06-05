"""
STEP 4 — utils.py
==================
Centralised save / load helpers for LDA results.

Any key added, removed, or renamed had to be updated in three places.
This module is now the single source of truth for the result schema.

Usage:
    from utils import save_lda_results, load_lda_results, unpack_run_result

    # After lda.run():
    result = lda.run(...)
    save_lda_results("./outputs/my_run.npz", result)

    # To reload:
    arrays = load_lda_results("./outputs/my_run.npz")
"""

import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Schema: ordered list of (key, result_index) pairs.
# result_index matches the position in the 21-value tuple returned by lda.run()
# ---------------------------------------------------------------------------
_RESULT_SCHEMA = [
    # key              index in run() return tuple
    ('word_score',       0),
    ('word_prob',        1),
    ('loss',             2),
    ('S',                3),
    ('topic',            4),
    ('rho',              5),
    ('Rho',              6),
    ('cntTW',            7),
    ('prR',              8),
    ('prFullCond',       9),
    ('phi',             10),
    ('phi_best',        11),
    ('attention_mask',  12),
    ('attention_mask2', 13),
    ('attention_mask3', 14),
    ('attention_mask4', 15),
    ('attention_mask5', 16),
    ('reward',          17),
    ('epoch_regret_history',  18),
    ('sweep_regret_history',  19),
    ('idx',             20),
]

RESULT_KEYS = [k for k, _ in _RESULT_SCHEMA]


def unpack_run_result(result: tuple) -> dict:
    """
    Convert the 21-value run() tuple into a named dictionary.

    Args:
        result: the tuple returned by lda.run()

    Returns:
        dict mapping key names to their array values
    """
    if len(result) != 21:
        raise ValueError(f"Expected 21-value result tuple, got {len(result)}")
    return {key: result[idx] for key, idx in _RESULT_SCHEMA}


def save_lda_results(path, result, prefix: str = '') -> Path:
    """
    Save the full output of lda.run() to a compressed .npz file.

    Args:
        path:   output file path (str or Path). '.npz' appended if missing.
        result: the 21-value tuple returned by lda.run(), OR a dict from
                unpack_run_result().
        prefix: optional string prepended to all keys (e.g. 'baseline_')

    Returns:
        Path to the written file.

    Example:
        save_lda_results('./outputs/run_400iter.npz', lda.run(...))
        save_lda_results('./outputs/run_400iter.npz', lda.run(...), prefix='mab_')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(result, tuple):
        data = unpack_run_result(result)
    elif isinstance(result, dict):
        data = result
    else:
        raise TypeError(f"result must be a tuple or dict, got {type(result)}")

    arrays = {f'{prefix}{k}': np.array(v) for k, v in data.items()}
    np.savez_compressed(path, **arrays)
    print(f'Saved {len(arrays)} arrays to {path}')
    return path


def load_lda_results(path, prefix: str = '') -> dict:
    """
    Load a .npz file saved by save_lda_results().

    Args:
        path:   path to .npz file
        prefix: prefix that was used when saving (stripped on load)

    Returns:
        dict mapping unprefixed key names to numpy arrays
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Result file not found: {path}')

    data = np.load(path, allow_pickle=True)
    result = {}
    for raw_key in data.files:
        clean_key = raw_key[len(prefix):] if raw_key.startswith(prefix) else raw_key
        result[clean_key] = data[raw_key]
    return result


def save_baseline_and_mab(output_dir, baseline_result, mab_result,
                           domain: str = '') -> tuple:
    """
    Convenience wrapper used for domain-specific/transfer experiments.
    Saves both baseline and MAB results with consistent naming.

    Args:
        output_dir:       directory for output files
        baseline_result:  21-value tuple from baseline LDA run
        mab_result:       21-value tuple from MAB-LDA run
        domain:           short domain label, e.g. 'aviation' or 'quantum'

    Returns:
        (baseline_path, mab_path)
    """
    suffix = f'_{domain}' if domain else ''
    bp = save_lda_results(
        Path(output_dir) / f'lda_results{suffix}_baseline.npz',
        baseline_result,
        prefix='baseline_'
    )
    mp = save_lda_results(
        Path(output_dir) / f'lda_results{suffix}.npz',
        mab_result,
        prefix='mab_'
    )
    return bp, mp
