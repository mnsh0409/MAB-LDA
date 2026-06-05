"""
extract_topics.py — extract interpretable topic words from MAB-LDA npz
Uses the Herfindahl importance filter to exclude stopwords and special tokens.
Same logic as the inference filter in eval_metrics.py.

Usage:
    python extract_topics.py \
        --npz_path outputs/semeval14_rest/lda_semeval14_rest_5k_400iter_gamma2.0.npz \
        --model_name_or_path bert-base-uncased \
        --gamma 2.05 \
        --top_n 10

"""
import argparse
import numpy as np
from transformers import AutoTokenizer

SPECIAL_IDS = {0, 100, 101, 102, 103}   # PAD, UNK, CLS, SEP, MASK

def extract_topics(npz_path, model_name, gamma, top_n):
    r = np.load(npz_path, allow_pickle=True)

    phi_best = r['phi_best']        # [K, V]  P(word|topic)
    Rho_arr  = r.get('Rho', None)   # [S_max, V]  HHI history
    S        = int(r['S'])
    K, V     = phi_best.shape

    tok = AutoTokenizer.from_pretrained(model_name)

    # ---- Build importance mask (same as eval_metrics.py) ----
    threshold = gamma / K
    if Rho_arr is not None:
        H_scores = Rho_arr[S]                           # [V]  HHI at best sweep
    else:
        # Fallback: compute HHI column-wise from phi_best
        phi_col  = phi_best / (phi_best.sum(axis=0, keepdims=True) + 1e-10)  # [K,V] → P(k|w)
        H_scores = np.sum(phi_col ** 2, axis=0)         # [V]

    # ---- Additional hard exclusions ----
    # Exclude PAD/CLS/SEP/UNK regardless of HHI score
    special_mask = np.zeros(V, dtype=bool)
    for sid in SPECIAL_IDS:
        if sid < V:
            special_mask[sid] = True

    # Exclude single-char punctuation tokens
    for wid in range(V):
        token = tok.convert_ids_to_tokens([wid])[0]
        if len(token) <= 1 or token in {'##', ',', '.', '!', '?', "'", '"', '-', '(', ')'}:
            special_mask[wid] = True

    important = (H_scores > threshold) & (~special_mask)

    n_important = important.sum()
    print(f'Vocabulary: {V:,}  |  Important (H>{threshold:.3f}): {n_important:,}  ({n_important/V:.1%})')
    print()

    # ---- Top words per topic from FILTERED vocabulary ----
    print(f'Top-{top_n} words per topic (γ={gamma}, γ_filter threshold={threshold:.3f}):')
    print()

    aspect_labels = {
        0: 'food/cuisine',
        1: 'service/staff',
        2: 'ambience/location',
        3: 'price/value',
        4: 'overall/anecdotes',
    }

    rows = []
    for k in range(K):
        phi_k = phi_best[k].copy()           # [V]  P(word|topic k)
        phi_k[~important] = -1               # mask out non-important words

        top_ids   = np.argsort(phi_k)[-top_n:][::-1]
        top_ids   = [i for i in top_ids if phi_k[i] > 0]   # remove masked
        top_words = tok.convert_ids_to_tokens(top_ids)

        # Clean ## subword prefix for readability
        top_words = [w.replace('##', '') if w.startswith('##') else w
                     for w in top_words]

        label = aspect_labels.get(k, f'topic {k}')
        print(f'  Topic {k} ({label}): {", ".join(top_words)}')
        rows.append((k, label, ', '.join(top_words)))

    print()
    print('LaTeX table row (for paper):')
    print()
    print('\\begin{table}[h]')
    print('\\centering')
    print('\\caption{Top-10 words per topic, SemEval-2014 Restaurant ($\\gamma{=}2.0$, K=5). '
          'Words selected by Herfindahl importance filter ($H(w)>{' + f'{threshold:.3f}' + '$).}}')
    print('\\label{tab:topics}')
    print('\\begin{tabular}{clp{6cm}}')
    print('\\toprule')
    print('Topic & Aspect & Top words \\\\')
    print('\\midrule')
    for k, label, words in rows:
        print(f'{k} & {label} & {words} \\\\')
    print('\\bottomrule')
    print('\\end{tabular}')
    print('\\end{table}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--npz_path',           required=True)
    p.add_argument('--model_name_or_path', default='bert-base-uncased')
    p.add_argument('--gamma',              type=float, default=2.05)
    p.add_argument('--top_n',              type=int,   default=10)
    args = p.parse_args()
    extract_topics(args.npz_path, args.model_name_or_path, args.gamma, args.top_n)
