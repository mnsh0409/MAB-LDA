"""
STEP 5 — run_domain.py
=======================
Usage via built-in domain:
    python run_domain.py aviation

Usage 2:
    python run_domain.py quantum

Usage (custom domain):
    python run_domain.py --csv path/to/data.csv \
                         --text_col abstract \
                         --anchors qubit quantum state algorithm \
                         --domain my_domain \
                         --num_docs 5000

The original scripts still work unchanged — this is an additive file.
"""

import argparse
import os
import sys
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from transformers import AutoTokenizer

# Import both model variants
from model_simulation_reward  import LDAGibbs as BaselineLDA   # Step 6 names
from model_exclusivity_reward import LDAGibbs as MABLDA
from utils import save_baseline_and_mab                         # Step 4
from utils_logging import SweepLogger


# ---------------------------------------------------------------------------
# Built-in domain configurations
# ---------------------------------------------------------------------------
DOMAIN_CONFIGS = {
    'yelp': {
        'csv_path':   None,              # loaded from HuggingFace streaming
        'sep':        None,
        'text_col':   'text',
        'col_names':  None,
        'anchors':    ['food', 'service', 'price', 'ambience'],
        'plot_title': 'Domain Transfer: Yelp Restaurant Reviews',
        'num_docs':   10_000,
    },
    'aviation': {
        # Download from Kaggle: search 'Top 10 Airlines Reviews 2016-2023'
        # Place at: data/Top10_airlines_reviews(2016-2023Jul).txt
        'csv_path':   '../data/Top10_airlines_reviews(2016-2023Jul).txt',
        'sep':        '\t',
        'text_col':   'full_text',           # constructed from ReviewTitle + ReviewText
        'col_names':  [
            'TopNumber', 'AirlineName', 'ReviewerName', 'Rating', 'ReviewDate',
            'ReviewTitle', 'ReviewText', 'Tags', 'DateofTravel', 'Aspects',
            'ResponserName', 'ResponseDate', 'ResponseText', 'ReviewerProfileUrl',
            'UserReviewLink', 'AirlineReviewLink', 'CrawlTime'
        ],
        'anchors':    ['seat', 'delay', 'ticket', 'staff'],
        'plot_title': 'Macro-Convergence on Noisy Corpora (Aviation)',
        'num_docs':   100_000,
    },
    'quantum': {
        'csv_path':   '../data/SOTA/arxiv_quant_ph_5000.csv',
        'sep':        ',',
        'text_col':   'abstract',
        'col_names':  None,                  # CSV has a header row
        'anchors':    ['qubit', 'quantum', 'state', 'algorithm'],
        'plot_title': 'Macro-Convergence on Sparse Scientific Corpora (arXiv: quant-ph)',
        'num_docs':   100,
    },
}

# Anchor labels map to SemEval slot names for human readability
ANCHOR_LABELS = ['food/physical', 'service/operational', 'price/value', 'ambience/personnel']


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def prepare_domain_data(
    csv_path: str,
    text_col: str,
    anchors: list,
    tokenizer_name: str = 'meta-llama/Llama-3.1-8B',
    num_docs: int = 1000,
    sep: str = ',',
    col_names=None,
    max_length: int = 256,
    seed: int = 42,
) -> tuple:
    """
    Load a domain CSV, tokenise the text column, and build the anchor token tensor.

    Returns:
        (data_tensor, tokens_tensor)  ready for LDAGibbs.__init__()
    """
    sys.stdout.reconfigure(encoding='utf-8')
    print(f'Loading tokenizer: {tokenizer_name}')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Yelp: load from HuggingFace streaming (csv_path=None signals this)
    if csv_path is None:
        from datasets import load_dataset
        print(f'Streaming Yelp from HuggingFace ({num_docs} docs)...')
        dataset = load_dataset('yelp_review_full', split='train', streaming=True)
        texts = []
        for row in dataset:
            texts.append(row['text'])
            if len(texts) >= num_docs: break
        random.seed(seed)
        random.shuffle(texts)
        texts = texts[:num_docs]
    else:
        print(f'Loading data from {csv_path!r}...')
        try:
            df = pd.read_csv(csv_path, sep=sep, header=None if col_names else 'infer',
                             names=col_names, on_bad_lines='warn')
        except Exception as e:
            print(f'Error reading data: {e}')
            raise

    if csv_path is not None:
        if text_col == 'full_text' and 'full_text' not in df.columns:
            df.dropna(subset=['ReviewText', 'ReviewDate', 'AirlineName'], inplace=True)
            df['full_text'] = df['ReviewTitle'].fillna('') + '. ' + df['ReviewText'].fillna('')
        texts_raw = df[text_col].fillna('').tolist()
        random.seed(seed)
        if len(texts_raw) > num_docs:
            texts = random.sample(texts_raw, num_docs)
        else:
            texts = texts_raw[:num_docs]

    print(f'Tokenising {len(texts)} documents...')
    encoded = tokenizer(texts, padding='max_length', truncation=True,
                        max_length=max_length, return_tensors='pt')
    data_tensor = encoded['input_ids']

    # Build anchor token tensor (same contract as original scripts)
    start_id = tokenizer.cls_token_id if tokenizer.cls_token_id else 0
    end_id   = tokenizer.sep_token_id if tokenizer.sep_token_id else 2
    pad_id   = tokenizer.pad_token_id if tokenizer.pad_token_id else 1
    in_id    = tokenizer.encode('in', add_special_tokens=False)[0]
    stop_id  = tokenizer.encode('.',  add_special_tokens=False)[0]

    anchor_ids = [tokenizer.encode(a, add_special_tokens=False)[0] for a in anchors]
    food_id, service_id, price_id, amb_id = anchor_ids

    anchor_tokens = [start_id, food_id, service_id, price_id, amb_id,
                     in_id, stop_id, end_id, pad_id]
    unique_tokens = torch.unique(data_tensor).tolist()
    rest          = [t for t in unique_tokens if t not in anchor_tokens]
    tokens_tensor = torch.tensor(anchor_tokens + rest)

    print(f'data_tensor: {data_tensor.shape}, tokens_tensor: {tokens_tensor.shape}')
    return data_tensor, tokens_tensor


# ---------------------------------------------------------------------------
# Convergence plot (was duplicated in both SOTA scripts)
# ---------------------------------------------------------------------------
def plot_convergence(baseline_loss, mab_loss, title: str, output_dir: str,
                     domain: str) -> None:
    clean_baseline = [l for l in baseline_loss if l < 0]
    clean_mab      = [l for l in mab_loss      if l < 0]

    plt.style.use('default')
    plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(clean_baseline,
            label=r'Baseline Standard Gibbs ($\phi$)',
            color='#E63946', linestyle='--', linewidth=2)
    ax.plot(clean_mab,
            label=r'Ours: MAB-LDA ($\rho$) w/ Exclusivity',
            color='#1D3557', linestyle='-',  linewidth=2.5)

    ax.set_title(title)
    ax.set_xlabel('MCMC Iterations')
    ax.set_ylabel(r'Log-Likelihood $\mathcal{L}$')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right')
    plt.tight_layout()

    for fmt in ('pdf', 'png'):
        out = os.path.join(output_dir, f'{domain}_convergence_plot.{fmt}')
        plt.savefig(out, format=fmt, bbox_inches='tight')
        print(f'Saved plot: {out}')
    plt.close()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Run MAB-LDA on a domain corpus')
    parser.add_argument('domain', nargs='?', choices=list(DOMAIN_CONFIGS.keys()),
                        help='Built-in domain name (aviation or quantum)')
    parser.add_argument('--csv',          type=str,  default=None)
    parser.add_argument('--text_col',     type=str,  default='abstract')
    parser.add_argument('--anchors',      nargs=4,   default=None,
                        metavar=('FOOD', 'SERVICE', 'PRICE', 'AMBIENCE'))
    parser.add_argument('--domain',       type=str,  dest='domain_name',
                        default='custom', help='Label for output files')
    parser.add_argument('--tokenizer',    type=str,  default='bert-base-uncased',
                        help='Tokenizer for domain experiments. '
                             'Paper results use bert-base-uncased.')
    parser.add_argument('--num_docs',     type=int,  default=None)
    parser.add_argument('--n_topics',     type=int,  default=8)
    parser.add_argument('--epochs',       type=int,  default=5)
    parser.add_argument('--iters',        type=int,  default=800)
    parser.add_argument('--burnin_ratio', type=float, default=0.0625)
    parser.add_argument('--output_dir',   type=str,  default='./outputs')
    args = parser.parse_args()

    # Resolve configuration
    if args.domain and args.domain in DOMAIN_CONFIGS:
        cfg = DOMAIN_CONFIGS[args.domain].copy()
        domain_name = args.domain
    else:
        if not args.csv or not args.anchors:
            parser.error('Provide a built-in domain name, or --csv and --anchors')
        cfg = {
            'csv_path':  args.csv,
            'sep':       ',',
            'text_col':  args.text_col,
            'col_names': None,
            'anchors':   args.anchors,
            'plot_title': f'Convergence ({args.domain_name})',
            'num_docs':  args.num_docs or 1000,
        }
        domain_name = args.domain_name

    if args.num_docs:
        cfg['num_docs'] = args.num_docs

    os.makedirs(args.output_dir, exist_ok=True)
    print(time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime()))

    data_tensor, tokens_tensor = prepare_domain_data(
        csv_path       = cfg['csv_path'],
        text_col       = cfg['text_col'],
        anchors        = cfg['anchors'],
        tokenizer_name = args.tokenizer,
        num_docs       = cfg['num_docs'],
        sep            = cfg['sep'],
        col_names      = cfg.get('col_names'),
    )

    # --- Baseline ---
    print('\n' + '='*50)
    print('RUNNING BASELINE (Standard Gibbs φ)')
    print('='*50)
    print(time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime()))
    baseline_lda = BaselineLDA(data_tensor, args.n_topics, tokens_tensor)
    baseline_lda._sweep_logger = SweepLogger(
        log_path = f'./logs/{domain_name}_baseline.csv',
        dataset  = domain_name,
        model    = 'Baseline-Gibbs',
        n_docs   = data_tensor.shape[0],
        n_topics = args.n_topics,
    )
    baseline_result = baseline_lda.run(
        epochs=args.epochs, max_iter=args.iters, burnin_ratio=args.burnin_ratio)
    baseline_lda._sweep_logger.close()
    baseline_loss = baseline_result[2]

    # --- MAB-LDA ---
    print('\n' + '='*50)
    print('RUNNING MAB-LDA (Thompson + Exclusivity ρ)')
    print('='*50)
    print(time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime()))
    mab_lda = MABLDA(data_tensor, args.n_topics, tokens_tensor)
    mab_lda._sweep_logger = SweepLogger(
        log_path = f'./logs/{domain_name}_mab.csv',
        dataset  = domain_name,
        model    = 'MAB-LDA',
        n_docs   = data_tensor.shape[0],
        n_topics = args.n_topics,
    )
    mab_result = mab_lda.run(
        epochs=args.epochs, max_iter=args.iters, burnin_ratio=args.burnin_ratio)
    mab_lda._sweep_logger.close()
    mab_loss = mab_result[2]
    print(time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime()))

    # --- Save ---
    save_baseline_and_mab(
        args.output_dir, baseline_result, mab_result, domain=domain_name)

    # --- Plot ---
    plot_convergence(baseline_loss, mab_loss,
                     title=cfg['plot_title'],
                     output_dir=args.output_dir,
                     domain=domain_name)
    print('Done.')


if __name__ == '__main__':
    main()
