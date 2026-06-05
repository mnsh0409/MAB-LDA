"""
model_exclusivity_reward.py  — ICDM main contribution
======================================================
MAB-LDA with exclusivity-score reward signal.

Reward: word-topic concentration (sparsity of word's topic distribution).
    word_topic_dist   = cntTW[:, w] / (cntW[w] + ε)
    exclusivity_score = Σ(word_topic_dist²)   [Herfindahl index]
    reward = 1 if exclusivity_score > (1/K) * gamma else 0

The gamma parameter controls strictness: higher gamma = fewer rewards =
more conservative topic assignments. Default 1.5 found to work well
across all ICDM benchmark datasets.

Storage access: uses CountStore API only — no direct array indexing.
This ensures the reward computation is correct regardless of whether
DenseCountStore, SparseCountStore, or CUDACountStore is the backend.
"""

import math
import numpy as np
from model_base import LDAGibbsBase


class LDAGibbs(LDAGibbsBase):
    """
    Exclusivity-reward MAB-LDA (sparsity model).
    Drop-in replacement for model_SemEval_sparsity.LDAGibbs.
    All constructor arguments and the 21-value return tuple are identical.
    """

    def _compute_reward(self, w_int: int, new_z: int,
                        prFullCond: np.ndarray) -> int:
        """
        Exclusivity-score (Herfindahl index) reward.

        Measures how exclusively a word belongs to a single topic.
        A perfectly exclusive word would have score = 1.0 (all mass on one topic).
        A uniform word would have score = 1/K.

        The threshold (1/K) * gamma separates exclusive from non-exclusive words.
        gamma > 1 requires more exclusivity than uniform → stricter.
        gamma < 1 accepts near-uniform words → more permissive.
        """
        epsilon = 1e-10
        # Use CountStore API — backend-agnostic
        tw_col          = self._store.get_tw_col(w_int)       # [K]
        cntW_w          = self._store.get_w(w_int)
        word_topic_dist = tw_col / (cntW_w + epsilon)
        excl_score      = float(np.sum(np.square(word_topic_dist)))
        return 1 if excl_score > (1.0 / self.TOPICS) * self.gamma else 0

    def _scale_dirichlet(self, prFullCond: np.ndarray,
                         epsilon: float) -> np.ndarray:
        """
        Dynamic concentration gate: scale = 1 + 100 * exp(L).

        As loss L → 0 (convergence), scale → 101 (sharply peaked Dirichlet).
        As loss L → -∞ (early training), scale → 1 (near-uniform Dirichlet).
        This implements the confidence-weighted exploration-exploitation trade-off.

        disable_gate=True ablates this to a flat scale for ablation studies.
        """
        if self.disable_gate:
            return prFullCond + epsilon
        return prFullCond * (1.0 + 100.0 * math.exp(self.loss)) + epsilon
