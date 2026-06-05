"""
eval_metrics.py
===============
Evaluation of MAB-LDA topic assignments against ground-truth aspect labels.

Computes the metrics reported in the ICDM paper tables:
  - Aspect classification F1 (macro, micro, weighted) per dataset
  - Accuracy of topic-to-aspect alignment via Hungarian matching
  - Purity and NMI for unsupervised topic quality
  - NPMI coherence for the 20 Newsgroups benchmark (TIT/ISIT paper)

Usage:
    python eval_metrics.py \
        --npz_path ./outputs/lda_semeval14_rest_5k_400iter.npz \
        --dataset  semeval14_rest \
        --data_path ../data/SemEval14/Restaurants_Test_Data_phaseB.xml \
        --model_name_or_path bert-base-uncased

    python eval_metrics.py \
        --npz_path ./outputs/lda_ng20_20k_400iter.npz \
        --dataset  ng20 \
        --model_name_or_path bert-base-uncased \
        --compute_npmi          # enables NPMI for ISIT/TIT
"""

import argparse
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    f1_score, accuracy_score, normalized_mutual_info_score,
    classification_report,
)
from utils import load_lda_results


# ---------------------------------------------------------------------------
# Aspect column definitions per dataset (mirrors dataload_SemEval.py)
# ---------------------------------------------------------------------------
DATASET_ASPECTS = {
    'semeval14_rest':   ['food', 'service', 'price', 'ambience', 'anecdotes/miscellaneous'],
    'semeval14_laptop': ['performance', 'design', 'usability', 'price', 'support'],
    'semeval15_rest':   ['food', 'service', 'price', 'ambience', 'restaurant'],
    'semeval16_rest':   ['food', 'service', 'price', 'ambience', 'restaurant'],
    'mams':             ['food', 'service', 'price', 'ambience',
                         'location', 'drinks', 'restaurant', 'miscellaneous'],
    'ng20':             [f'topic_{i}' for i in range(20)],
}


# ===========================================================================
# 1. Topic-to-aspect alignment via Hungarian matching
# ===========================================================================
def hungarian_align(pred_topics: np.ndarray,
                    true_labels: np.ndarray,
                    n_topics: int,
                    n_aspects: int) -> np.ndarray:
    """
    Find the optimal 1-to-1 mapping from predicted topic IDs to
    ground-truth aspect IDs using the Hungarian algorithm.

    Args:
        pred_topics:  [N] predicted topic assignment per document
        true_labels:  [N] ground-truth aspect label per document
        n_topics:     K — number of predicted topics
        n_aspects:    number of ground-truth aspects

    Returns:
        mapping: np.ndarray [K] where mapping[k] = best matching aspect id
    """
    # Build cost matrix [K, A]: count how often topic k is assigned to aspect a
    cost = np.zeros((n_topics, n_aspects), dtype=int)
    for k in range(n_topics):
        mask = pred_topics == k
        if mask.sum() == 0:
            continue
        for a in range(n_aspects):
            cost[k, a] = np.sum(true_labels[mask] == a)

    # Maximise overlap (negate for minimization)
    row_ind, col_ind = linear_sum_assignment(-cost)
    mapping = np.zeros(n_topics, dtype=int)
    for r, c in zip(row_ind, col_ind):
        mapping[r] = c
    return mapping


# ===========================================================================
# 2. Purity
# ===========================================================================
def cluster_purity(pred_topics: np.ndarray,
                   true_labels: np.ndarray) -> float:
    """
    Purity = (1/N) Σ_k max_a |cluster_k ∩ class_a|
    Standard unsupervised clustering quality metric.
    Reported in ICDM Table 2 alongside NMI.
    """
    n = len(pred_topics)
    total = 0
    for k in np.unique(pred_topics):
        mask = pred_topics == k
        if mask.sum() == 0:
            continue
        counts = np.bincount(true_labels[mask].astype(int),
                             minlength=int(true_labels.max()) + 1)
        total += counts.max()
    return total / n


# ===========================================================================
# 3. NPMI coherence (ISIT / TIT metric)
# ===========================================================================
def compute_npmi(phi: np.ndarray,
                 corpus_ids: np.ndarray,
                 top_n: int = 10,
                 epsilon: float = 1e-12) -> float:
    """
    Normalised Pointwise Mutual Information (NPMI) coherence.
    The standard metric for topic model quality in the information theory
    and NLP literature. Used in the ISIT paper as the observable M
    in the classical shadow complexity theorem.

    NPMI(w_i, w_j) = log[p(w_i, w_j) / (p(w_i) * p(w_j))]
                     / -log[p(w_i, w_j)]

    Averaged over all top-N word pairs per topic, then over all topics.

    Args:
        phi:        [K, V] topic-word distribution (normalised rows)
        corpus_ids: [D, L] token ID matrix (raw corpus, not just top words)
        top_n:      number of top words per topic to use
        epsilon:    smoothing for log-zero

    Returns:
        mean NPMI across all topics and word pairs (higher is better,
        range [-1, 1], values > 0.1 considered good)

    Publication note:
        ICDM: report this alongside F1 for 20NG experiments
        ISIT: this is the key quantity in the information-ratio analysis
        TIT:  this is the observable O in the classical shadow theorem
    """
    D, L = corpus_ids.shape
    V    = phi.shape[1]
    K    = phi.shape[0]

    # Build boolean doc-word matrix for the unique words needed
    # Mimno et al. (2011) NPMI uses DOCUMENT-LEVEL probabilities:
    #   p(w)      = |{d : w in d}| / D
    #   p(w_i,w_j)= |{d : w_i in d AND w_j in d}| / D
    # Both denominators are D (not D*L). This is the standard.
    # Bug was: p_w used N_tokens=D*L while p_wiwj used D -> huge PMI.

    # Collect all unique top-N words across topics
    all_top = set()
    top_per_topic = []
    for k in range(K):
        tw = np.argsort(phi[k])[-top_n:]
        top_per_topic.append(tw)
        all_top.update(tw.tolist())

    # Precompute doc presence for each needed word [len(all_top), D]
    all_top_list = sorted(all_top)
    word_to_idx  = {w: i for i, w in enumerate(all_top_list)}
    presence     = np.zeros((len(all_top_list), D), dtype=bool)
    for i, w in enumerate(all_top_list):
        presence[i] = np.any(corpus_ids == w, axis=1)  # [D] bool

    # doc_freq[w] = number of documents containing w
    doc_freq = presence.sum(axis=1).astype(float)      # [len(all_top)]

    npmi_scores = []
    for k in range(K):
        top_words  = top_per_topic[k]
        topic_npmi = []
        for i in range(len(top_words)):
            for j in range(i + 1, len(top_words)):
                w_i = top_words[i]
                w_j = top_words[j]
                ii  = word_to_idx[w_i]
                jj  = word_to_idx[w_j]

                # All document-level probabilities
                p_wi   = doc_freq[ii] / D
                p_wj   = doc_freq[jj] / D
                p_wiwj = np.sum(presence[ii] & presence[jj]) / D

                if p_wiwj < epsilon or p_wi < epsilon or p_wj < epsilon:
                    continue

                pmi  = np.log(p_wiwj / (p_wi * p_wj))
                npmi = pmi / (-np.log(p_wiwj + epsilon))
                topic_npmi.append(float(npmi))

        if topic_npmi:
            npmi_scores.append(np.mean(topic_npmi))

    return float(np.mean(npmi_scores)) if npmi_scores else 0.0


# ===========================================================================
# 4. Information ratio of Thompson sampler (ISIT-specific metric)
# ===========================================================================
def compute_information_ratio(reward_history: np.ndarray,
                               rho: np.ndarray,
                               epsilon: float = 1e-10) -> float:
    """
    Information ratio Γ_t = (Δ_t)² / g_t

    where:
        Δ_t = expected regret at step t = prFullCond.max() - prFullCond[new_z]
        g_t = information gain = KL(posterior || prior) for the selected topic

    Used in the ISIT paper's information-theoretic analysis of Thompson sampling
    applied to the exclusivity reward signal.

    Reference: Russo & Van Roy (2016) "An Information-Theoretic Analysis of
    Thompson Sampling" — the framework we apply to MAB-LDA.

    Args:
        reward_history: [V] cumulative reward per vocabulary word
        rho:            [V, K] MAB reward accumulation matrix

    Returns:
        estimated information ratio (lower is better — tighter regret bound)

    Publication note:
        ISIT paper: bound E[R_T] ≤ sqrt(T * K * Γ) using this ratio
        TIT paper:  prove Γ ≤ O(log K / exclusivity_threshold) formally
    """
    # Per-word information gain: entropy of rho distribution
    rho_norm = rho / (rho.sum(axis=1, keepdims=True) + epsilon)  # [V, K]

    # Shannon entropy H(rho[w]) as proxy for information gain per word
    entropy = -np.sum(rho_norm * np.log(rho_norm + epsilon), axis=1)  # [V]

    # Expected regret proxy: words with low reward have high regret
    total_reward = reward_history.sum()
    if total_reward < epsilon:
        return float('inf')

    # Information ratio estimate: regret² / information_gain
    word_regret = 1.0 - (reward_history / (reward_history.max() + epsilon))
    valid = entropy > epsilon
    if valid.sum() == 0:
        return float('inf')

    ratio = np.mean((word_regret[valid] ** 2) / (entropy[valid] + epsilon))
    return float(ratio)


# ===========================================================================
# 5. Full evaluation pipeline
# ===========================================================================
def evaluate(npz_path: str,
             dataset: str,
             data_path: str = None,
             model_name_or_path: str = 'bert-base-uncased',
             compute_npmi_flag: bool = False,
             top_n_words: int = 10,
             prefix: str = '',
             gamma: float = 1.5,
             importance_signal: str = 'Rho',
             filter_mode: str = 'hard') -> dict:
    """
    Full evaluation pipeline. Loads saved .npz results and computes
    all metrics appropriate for the given dataset and target venue.

    Args:
        npz_path:           path to saved lda_*.npz results file
        dataset:            dataset name from DATASET_ASPECTS keys
        data_path:          path to original XML/CSV (for label loading)
        model_name_or_path: tokenizer for reloading aspect labels
        compute_npmi_flag:  enable NPMI computation (slower, needs corpus)
        top_n_words:        top words per topic for NPMI
        prefix:             key prefix used when saving (e.g. 'mab_')

    Returns:
        dict of all computed metrics
    """
    print(f'\n{"="*60}')
    print(f'Evaluating: {dataset}  |  {npz_path}')
    print(f'{"="*60}')

    results = load_lda_results(npz_path, prefix=prefix)
    aspect_cols = DATASET_ASPECTS.get(dataset, [])
    n_aspects   = len(aspect_cols)

    # Extract key arrays
    phi         = results.get('phi_best', None)
    rho         = results.get('rho',         None)
    Rho_arr     = results.get('Rho',         None)
    prR_arr     = results.get('prR',         None)
    pfc_arr     = results.get('prFullCond',  None)
    phi_hist    = results.get('phi',         None)
    reward      = results.get('reward',      None)
    topic_hist  = results.get('topic',       None)
    S           = int(results.get('S', 0))
    loss        = results.get('loss',        [])

    metrics = {
        'dataset':   dataset,
        'npz_path':  npz_path,
        'S':         S,
        'final_loss': float(loss[-1]) if len(loss) > 0 else None,
    }

    # --- Loss convergence summary ---
    real_losses = [l for l in loss if isinstance(l, (int, float)) and l < 0]
    if real_losses:
        metrics['loss_initial'] = real_losses[0]
        metrics['loss_final']   = real_losses[-1]
        metrics['loss_delta']   = real_losses[-1] - real_losses[0]
        print(f'Loss:  initial={real_losses[0]:.4f}  '
              f'final={real_losses[-1]:.4f}  '
              f'delta={real_losses[-1]-real_losses[0]:.4f}')

    # --- Reward summary ---
    if reward is not None:
        total_reward = float(reward.sum())
        metrics['total_reward']  = total_reward
        metrics['reward_rate']   = float(np.mean(reward > 0))
        print(f'Reward: total={total_reward:.0f}  '
              f'rate={metrics["reward_rate"]:.3f}')

    # --- Information ratio (ISIT metric) ---
    if rho is not None and reward is not None:
        ir = compute_information_ratio(reward, rho)
        metrics['information_ratio'] = ir
        print(f'Information ratio Γ: {ir:.6f}  '
              f'(ISIT metric — lower is better)')

    # --- Supervised metrics: load ground truth if data_path provided ---
    if data_path and dataset != 'ng20' and aspect_cols:
        try:
            from dataload_SemEval import (
                get_semeval_data, get_semeval14_laptop,
                get_semeval15_data, get_semeval16_data, get_mams_data
            )
            loaders = {
                'semeval14_rest':   get_semeval_data,
                'semeval14_laptop': get_semeval14_laptop,
                'semeval15_rest':   get_semeval15_data,
                'semeval16_rest':   get_semeval16_data,
                'mams':             get_mams_data,
            }
            loader = loaders.get(dataset)
            if loader:
                df, _, _ = loader(data_path, model_name_or_path)
                _supervised_metrics(df, aspect_cols, topic_hist, rho,
                                    metrics, n_aspects,
                                    Rho_arr=Rho_arr, S=S, phi=phi, gamma=gamma,
                                    prR_arr=prR_arr, pfc_arr=pfc_arr,
                                    phi_hist=phi_hist, importance_signal=importance_signal,
                                    filter_mode=filter_mode,
                                    dataset=dataset)
        except Exception as e:
            print(f'Warning: could not load ground truth labels: {e}')

    # --- NPMI coherence (ISIT / TIT / 20NG experiments) ---
    if compute_npmi_flag and phi is not None:
        print('\nComputing NPMI coherence...')
        try:
            # phi = phi_best [K, V] — correct shape guaranteed by loader above
            # The original bug was phi=[400,V] (sweep history); now phi=phi_best [K,V]
            if phi.ndim == 2 and phi.shape[0] <= 100:
                phi_matrix = phi          # [K, V] topic-word matrix
            else:
                raise ValueError(f'phi shape {phi.shape} unexpected for NPMI')

            # Reload corpus token IDs for co-occurrence
            # word_score in npz = lda_embeddings (float), NOT token IDs
            # Must reload from dataset loader
            if dataset == 'ng20':
                from dataload_SemEval import get_20newsgroups
                df_ng, _, _ = get_20newsgroups(model_name_or_path, max_docs=5000)
                corpus_ids = np.array(list(df_ng['input_ids']))
            else:
                raise ValueError(f'NPMI not supported for dataset={dataset}')

            npmi_score = compute_npmi(phi_matrix, corpus_ids,
                                      top_n=top_n_words)
            metrics['npmi'] = round(float(npmi_score), 4)
            print(f'  NPMI coherence: {npmi_score:.4f}  '
                  f'(ISIT/TIT metric — >0.1 is good)')
        except Exception as e:
            print(f'Warning: NPMI computation failed: {e}')
            metrics['npmi'] = None

    # --- Print full summary ---
    print(f'\n--- Metrics summary ({dataset}) ---')
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f'  {k:<30}: {v:.4f}')
        elif isinstance(v, (int, np.integer)):
            print(f'  {k:<30}: {v}')

    return metrics


def _supervised_metrics(df, aspect_cols, topic_hist, rho,
                         metrics, n_aspects,
                         Rho_arr=None, S=0, phi=None, gamma=1.5,
                         prR_arr=None, pfc_arr=None, phi_hist=None,
                         importance_signal='Rho', filter_mode='hard',
                         dataset=''):
    """Compute Hungarian-aligned F1, accuracy, purity, NMI."""
    # Build document-level aspect label (majority voting across aspects)
    # For binary aspects: pick the aspect with highest polarity confidence
    aspect_matrix = df[aspect_cols].replace(2, np.nan).values  # [D, A]

    # Per-document predicted topic using topic_hist (argmax of prFullCond per word).
    # WHY NOT rho: rho[w,:] = [0,0,0,0,0] for ~93% of words (low reward rate).
    #   argmax([0,...,0]) = 0 always → every document predicted as topic 0.
    # topic_hist is set for ALL vocabulary words at every sweep via
    #   topic_hist[s] = argmax(self.prFullCond, axis=1), reflecting the
    #   full Gibbs posterior — the correct signal for classification.
    input_ids = np.array(list(df['input_ids']))  # [D, L]
    D = len(df)
    K = n_aspects

    # Root cause of F1=0.52 (all-food prediction):
    # max_length=512 but avg SemEval sentence = 20 tokens → 96% PAD.
    # topic_hist[PAD_ID] = 0 (argmax of uniform prFullCond = first index).
    # 492 PAD votes overwhelm 20 real votes → every doc predicted as topic 0.
    # FIX: use df['attention_mask'] to exclude PAD/CLS/SEP before majority vote.
    # attention_mask[d, pos] = 1 for real tokens, 0 for padding (from tokenizer).
    attn_masks = None
    if 'attention_mask' in df.columns:
        try:
            attn_masks = np.array(list(df['attention_mask']))  # [D, L]
        except Exception:
            attn_masks = None

    # Build Herfindahl importance mask from Rho[S]
    # Original design intent (model_base.py lines 508-510):
    #   Rho[s, w] = Σ_k (rho[w,k] / Σ_k rho[w,k])²  = HHI of word w at sweep s
    #   Rho[S, w] = HHI at the BEST sweep S — already precomputed
    # This is the canonical importance score from the MAB reward signal.
    # Only exclusive tokens (H > γ/K) vote for document topic.
    # Noise tokens (stopwords, subwords) have H ≈ 1/K and are silenced.
    # S_int for H_scores lookup in prediction block below
    S_int = int(S)

    # SOFT HERFINDAHL WEIGHTING — Option B
    # Theoretical motivation: Proposition 1 (ρ*=Φᵀ) establishes that
    # Rho[S][w] = H_rho(w) = Σ_k p(k|w)² converges to H_phi(w) at the
    # variational optimum. Using Rho[S][w] directly as a vote weight:
    #   - eliminates the binary threshold (no γ/K cutoff needed)
    #   - exclusive words (H≈1.0) dominate the prediction
    #   - noise words (H≈1/K≈0.20) contribute minimally but not zero
    #   - same criterion as the reward signal, applied to inference
    # This is the most principled prediction method in the MAB-LDA framework.
    sig = importance_signal or 'Rho'
    H_scores = None
    if   sig=='Rho' and Rho_arr is not None and 0<=S_int<len(Rho_arr):
        H_scores = Rho_arr[S_int];  print(f'  Signal: Rho[S={S_int}]')
    elif sig=='prR' and prR_arr is not None and S_int<len(prR_arr):
        H_scores = prR_arr[S_int];  print(f'  Signal: prR[S={S_int}]')
    elif sig=='prFullCond' and pfc_arr is not None and S_int<len(pfc_arr):
        H_scores = pfc_arr[S_int];  print(f'  Signal: prFullCond[S={S_int}]')
    elif sig=='phi' and phi_hist is not None and S_int<len(phi_hist):
        H_scores = phi_hist[S_int]; print(f'  Signal: phi[S={S_int}]')
    elif phi is not None and phi.ndim==2 and phi.shape[0]==K:
        phi_col  = phi/(phi.sum(axis=0,keepdims=True)+1e-10)
        H_scores = np.sum(phi_col**2, axis=0); print('  Signal: phi_best (fallback)')

    pred_doc_topics = np.zeros(D, dtype=int)
    if topic_hist is not None and len(topic_hist) > 0:
        topic_per_word = topic_hist.astype(int)   # [V] dominant topic per word
        for d in range(D):
            word_ids = input_ids[d].astype(int)
            # Exclude PAD/CLS/SEP tokens
            if attn_masks is not None:
                real = attn_masks[d].astype(bool)
            else:
                real = ~np.isin(word_ids, [0, 101, 102])
            word_ids_real = np.clip(word_ids[real], 0, len(topic_per_word) - 1)
            if len(word_ids_real) == 0:
                continue
            if H_scores is not None and filter_mode == 'hard':
                # HARD filter: binary threshold — noise words completely silenced
                threshold = gamma / K if gamma else 1.0 / K
                excl = H_scores[word_ids_real] > threshold
                ids  = word_ids_real[excl] if excl.any() else word_ids_real
                word_topics = topic_per_word[ids]
                pred_doc_topics[d] = int(np.bincount(word_topics, minlength=K).argmax())
            elif H_scores is not None and filter_mode == 'soft':
                # SOFT weighting: vote weight = H_scores[w] (no threshold)
                scores = np.zeros(K)
                for w in word_ids_real:
                    scores[topic_per_word[w]] += H_scores[w]
                pred_doc_topics[d] = int(scores.argmax())
            else:
                # Fallback: uniform voting
                word_topics = topic_per_word[word_ids_real]
                pred_doc_topics[d] = int(np.bincount(word_topics, minlength=K).argmax())
    elif rho is not None:
        # Fallback: rho with attention mask
        for d in range(D):
            word_ids = np.clip(input_ids[d].astype(int), 0, rho.shape[0]-1)
            if attn_masks is not None:
                real = attn_masks[d].astype(bool)
                word_ids = word_ids[real]
            else:
                word_ids = word_ids[~np.isin(word_ids, [0, 101, 102])]
            wt = rho[word_ids]
            nonzero = wt.sum(axis=1) > 0
            if nonzero.any():
                topics = np.argmax(wt[nonzero], axis=1)
                pred_doc_topics[d] = int(np.bincount(topics, minlength=K).argmax())

    # Ground truth: document's dominant aspect.
    # Strategy: take the aspect column with non-NaN value.
    # Include neutral/conflict (polarity=2) — SemEval test set uses conflict
    # annotations which previously caused false "no valid labels" warnings.
    # raw aspect_matrix already has 2 replaced with NaN via .replace(2, np.nan)
    # so we restore: use df values directly without replacing 2
    aspect_matrix_full = df[aspect_cols].values  # [D, A] — keep 2s
    true_doc_aspects = np.full(D, -1, dtype=int)
    for d in range(D):
        for a in range(len(aspect_cols)):
            val = aspect_matrix_full[d, a]
            if not (isinstance(val, float) and np.isnan(val)):
                true_doc_aspects[d] = a   # first annotated aspect (any polarity)
                break

    valid = true_doc_aspects >= 0
    n_valid = int(valid.sum())
    if n_valid < 2:
        print(f'Warning: only {n_valid} documents with valid aspect labels — skipping supervised metrics.')
        print('  Try passing --data_path to the TRAIN file which has complete polarity annotations.')
        return
    if n_valid < 10:
        print(f'Note: only {n_valid} valid labels — metrics may be unreliable.')

    pred_v = pred_doc_topics[valid]
    true_v = true_doc_aspects[valid]

    # Hungarian alignment
    mapping = hungarian_align(pred_v, true_v, K, n_aspects)
    pred_aligned = np.array([mapping[t] for t in pred_v])

    # Metrics
    f1_macro   = f1_score(true_v, pred_aligned, average='macro',   zero_division=0)
    f1_micro   = f1_score(true_v, pred_aligned, average='micro',   zero_division=0)
    f1_weighted= f1_score(true_v, pred_aligned, average='weighted',zero_division=0)
    acc        = accuracy_score(true_v, pred_aligned)
    purity     = cluster_purity(pred_v, true_v)
    nmi        = normalized_mutual_info_score(true_v, pred_v)

    metrics.update({
        'f1_macro':    f1_macro,
        'f1_micro':    f1_micro,
        'f1_weighted': f1_weighted,
        'accuracy':    acc,
        'purity':      purity,
        'nmi':         nmi,
        'n_valid_docs': int(valid.sum()),
    })

    print(f'\nSupervised metrics ({valid.sum()} docs with labels):')
    print(f'  F1 macro:    {f1_macro:.4f}')
    print(f'  F1 micro:    {f1_micro:.4f}')
    print(f'  F1 weighted: {f1_weighted:.4f}')
    print(f'  Accuracy:    {acc:.4f}')
    print(f'  Purity:      {purity:.4f}')
    print(f'  NMI:         {nmi:.4f}')

    # Per-class report (for Table VIII in paper)
    from sklearn.metrics import classification_report
    aspect_names = {
        'semeval14_rest':   ['food','service','price','ambience','anecdote'],
        'semeval14_laptop': ['performance','design','usability','price','support'],
        'semeval15_rest':   ['food','service','price','ambience','anecdote'],
        'semeval16_rest':   ['food','service','price','ambience','anecdote'],
        'mams':             ['food','service','price','ambience','location',
                             'menu','staff','overall'],
    }
    names = aspect_names.get(dataset, [str(i) for i in range(n_aspects)])
    print('\nPer-class report (after Hungarian alignment):')
    print(classification_report(
        true_v, pred_aligned,
        target_names=names[:n_aspects],
        zero_division=0))
    print()


# ===========================================================================
# CLI
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(description='MAB-LDA evaluation metrics')
    p.add_argument('--npz_path',            type=str, required=True)
    p.add_argument('--dataset',             type=str, required=True,
                   choices=list(DATASET_ASPECTS.keys()))
    p.add_argument('--data_path',           type=str, default=None)
    p.add_argument('--model_name_or_path',  type=str, default='bert-base-uncased')
    p.add_argument('--compute_npmi',        action='store_true',
                   help='Compute NPMI coherence (needed for ISIT/TIT paper)')
    p.add_argument('--top_n_words',         type=int, default=10)
    p.add_argument('--gamma',               type=float, default=1.5,
                   help='Herfindahl threshold multiplier (default: 1.5)')
    p.add_argument('--importance_signal',   type=str, default='Rho',
                   choices=['Rho','prR','prFullCond','phi'],
                   help='Vote weight signal: Rho|prR|prFullCond|phi (default: Rho)')
    p.add_argument('--filter_mode',          type=str, default='hard',
                   choices=['hard','soft'],
                   help='hard=binary threshold, soft=continuous weight (default: hard)')
    p.add_argument('--prefix',              type=str, default='',
                   help="Key prefix used when saving (e.g. 'mab_')")
    p.add_argument('--output_csv',          type=str, default=None,
                   help='Optional CSV path to append metrics row')
    return p.parse_args()


def main():
    args = parse_args()
    # Expand glob pattern (e.g. outputs/run/*.npz) to the first matching file
    npz_path = args.npz_path
    if '*' in npz_path or '?' in npz_path:
        matches = sorted(glob.glob(npz_path))
        if not matches:
            raise FileNotFoundError(
                f'No files matched pattern: {npz_path}\n'
                f'Run ls {npz_path} in bash to check the path.')
        npz_path = matches[0]
        if len(matches) > 1:
            print(f'Multiple matches — using first: {npz_path}')

    metrics = evaluate(
        npz_path            = npz_path,
        dataset             = args.dataset,
        data_path           = args.data_path,
        model_name_or_path  = args.model_name_or_path,
        compute_npmi_flag   = args.compute_npmi,
        top_n_words         = args.top_n_words,
        prefix              = args.prefix,
        gamma               = args.gamma,
        importance_signal   = args.importance_signal,
        filter_mode         = args.filter_mode,
    )
    if args.output_csv:
        row = pd.DataFrame([metrics])
        path = Path(args.output_csv)
        if path.exists():
            row.to_csv(path, mode='a', header=False, index=False)
        else:
            row.to_csv(path, index=False)
        print(f'Metrics appended to {args.output_csv}')


if __name__ == '__main__':
    main()
