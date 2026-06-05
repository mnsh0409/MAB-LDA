"""
extract_domain_topics.py — Reproduce Table X (Aviation Domain Transfer)
=========================================================================
Prints top-8 words per topic for MAB-LDA and Standard Gibbs on any
unlabelled corpus (no aspect annotations required).

Table X in the ICDM paper was generated with:

    python extract_domain_topics.py \
        --mab_npz    outputs/lda_aviation_mab.npz \
        --base_npz   outputs/lda_aviation_baseline.npz \
        --top_n      8

Topic words are shown after removing sklearn ENGLISH_STOP_WORDS for
readability ONLY — MAB-LDA trains on the full BERT vocabulary without
any stopword removal (Table X footnote †).

If you only have the MAB-LDA npz (no baseline), omit --base_npz.
"""
import argparse
import numpy as np
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

SPECIAL = {'[pad]', '[cls]', '[sep]', '[unk]', ''}


def top_words(phi_row: np.ndarray, tokenizer, n: int = 8) -> list:
    """Return top-n readable words for one topic row of phi."""
    top_ids = phi_row.argsort()[::-1]
    words = []
    for i in top_ids:
        w = tokenizer.decode([int(i)]).strip()
        w_lower = w.lower()
        if (w_lower not in SPECIAL
                and not w.startswith('##')
                and len(w) > 2
                and w_lower not in ENGLISH_STOP_WORDS):
            words.append(w)
        if len(words) >= n:
            break
    return words


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mab_npz',            required=True,
                   help='Path to MAB-LDA output .npz (from run_all.py)')
    p.add_argument('--base_npz',           default=None,
                   help='Path to Standard Gibbs baseline .npz (optional)')
    p.add_argument('--model_name_or_path', default='bert-base-uncased')
    p.add_argument('--top_n',              type=int, default=8)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # ── MAB-LDA topics ────────────────────────────────────────────────────
    mab_data = np.load(args.mab_npz, allow_pickle=True)

    # Accept both key names for backward compatibility
    if 'phi_best' in mab_data:
        phi_mab = mab_data['phi_best']          # run_all.py standard key
    elif 'mab_phi_best' in mab_data:
        phi_mab = mab_data['mab_phi_best']      # legacy key
    else:
        raise KeyError(f"No phi_best or mab_phi_best in {args.mab_npz}. "
                       f"Available keys: {list(mab_data.keys())}")

    K = phi_mab.shape[0]
    print(f'MAB-LDA  |  {K} topics, vocab={phi_mab.shape[1]}')
    print(f'(† topic words shown after sklearn stopword removal for readability;')
    print(f'   MAB-LDA trains on the full BERT vocabulary without stopword removal)\n')

    print('MAB-LDA Aviation Topics:')
    print('-' * 60)
    for k in range(K):
        words = top_words(phi_mab[k], tok, args.top_n)
        print(f'  Topic {k}: {", ".join(words)}')

    # ── Standard Gibbs baseline (optional) ───────────────────────────────
    if args.base_npz:
        print()
        base_data = np.load(args.base_npz, allow_pickle=True)

        if 'phi_best' in base_data:
            phi_base = base_data['phi_best']
        elif 'baseline_phi_best' in base_data:
            phi_base = base_data['baseline_phi_best']
        else:
            raise KeyError(f"No phi_best or baseline_phi_best in {args.base_npz}. "
                           f"Available keys: {list(base_data.keys())}")

        print(f'Standard Gibbs  |  {phi_base.shape[0]} topics, vocab={phi_base.shape[1]}')
        print('(no Herfindahl filter — topics dominated by punctuation/stopwords)\n')
        print('Standard Gibbs Aviation Topics:')
        print('-' * 60)
        for k in range(phi_base.shape[0]):
            words = top_words(phi_base[k], tok, args.top_n)
            print(f'  Topic {k}: {", ".join(words)}')


if __name__ == '__main__':
    main()
