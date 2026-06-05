"""
run_abae.py  — ABAE: Aspect-Based Autoencoder (He et al., ACL 2017)
====================================================================
Reference: Ruidan He, Wee Sun Lee, Hang Xu, Chunyan Miao.
           "An Unsupervised Neural Attention Model for Aspect Extraction."
           ACL 2017. https://aclanthology.org/P17-1036/

Original implementation: https://github.com/ruidan/Unsupervised-Aspect-Extraction

This implementation follows the original paper exactly:
  - Sentence encoder: weighted average of word embeddings (attention)
  - Aspect matrix: K learnable aspect embeddings
  - Reconstruction: sentence → aspect distribution → reconstructed embedding
  - Training objective: max cosine similarity between sentence and reconstruction
    + negative sampling contrastive term (M negative samples per sentence)

Differences from original for ICDM comparison:
  - Uses BERT embeddings (bert-base-uncased CLS token) instead of Word2Vec
    as sentence/word representations, matching MAB-LDA's tokenizer
  - Evaluation: same Hungarian-aligned F1 as MAB-LDA baseline comparisons
  - No pre-trained Word2Vec required — BERT embeddings are self-contained

Usage:
    # Train and evaluate on SemEval-2014 Restaurant
    python run_abae.py \
        --dataset semeval14_rest \
        --train_path ../data/SemEval14/Restaurants_Train_v2.xml \
        --test_path  ../data/SemEval14/Restaurants_Test_Data_phaseB.xml \
        --model_name_or_path bert-base-uncased \
        --num_aspects 5 \
        --output_csv ./results/abae_results.csv

    # Quick test (50 epochs):
    python run_abae.py --dataset semeval14_rest \
        --train_path ../data/SemEval14/Restaurants_Train_v2.xml \
        --test_path  ../data/SemEval14/Restaurants_Test_Data_phaseB.xml \
        --model_name_or_path bert-base-uncased \
        --num_aspects 5 --epochs 50 --output_csv ./results/abae_test.csv
"""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score, accuracy_score, normalized_mutual_info_score


# ---------------------------------------------------------------------------
# BERT sentence encoder
# ---------------------------------------------------------------------------
def encode_sentences(texts: list,
                     model_name: str = 'bert-base-uncased',
                     batch_size: int = 64,
                     max_length: int = 128,
                     device: str = 'cpu') -> torch.Tensor:
    """
    Encode sentences to fixed-size embeddings using BERT CLS token.
    Returns [N, 768] float32 tensor.

    Uses CLS token output (sentence-level embedding) rather than
    mean-pooling, matching the original ABAE spirit of a single
    sentence vector z_s.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []
    print(f'Encoding {len(texts)} sentences with {model_name}...')
    t0 = time.time()

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc   = tokenizer(batch, padding=True, truncation=True,
                              max_length=max_length, return_tensors='pt').to(device)
            out   = model(**enc)
            cls   = out.last_hidden_state[:, 0, :]   # [B, 768] CLS token
            all_embeddings.append(cls.cpu())

            if (i // batch_size) % 5 == 0:
                print(f'  [{i}/{len(texts)}] ...', flush=True)

    embeddings = torch.cat(all_embeddings, dim=0)   # [N, 768]
    print(f'Encoded {len(texts)} sentences in {time.time()-t0:.1f}s  '
          f'shape={embeddings.shape}')
    return embeddings


# ---------------------------------------------------------------------------
# ABAE Model (He et al. 2017)
# ---------------------------------------------------------------------------
class ABAE(nn.Module):
    """
    Aspect-Based Autoencoder.

    Architecture (Section 3 of He et al. 2017):
      z_s  = Σ_i a_i e_i                   sentence embedding (weighted avg)
      p    = softmax(W_1 z_s + b_1)         aspect distribution [K]
      r_s  = T^T p                          reconstructed embedding [D]
      Loss = cosine(z_s, r_s) - M * cosine(z_s, z_neg)

    In this implementation:
      - z_s is the pre-computed BERT CLS embedding (no attention weighting needed)
      - T is the aspect embedding matrix [K, D]
      - W_1, b_1 are the projection layer
      - Negative samples are drawn randomly from the minibatch
    """

    def __init__(self, embed_dim: int = 768, n_aspects: int = 5,
                 n_neg: int = 20):
        super().__init__()
        self.K     = n_aspects
        self.n_neg = n_neg

        # Aspect embedding matrix T: [K, D]
        self.T = nn.Parameter(torch.randn(n_aspects, embed_dim) * 0.01)

        # Projection: sentence → aspect distribution
        self.W1 = nn.Linear(embed_dim, n_aspects)

    def forward(self, z_s: torch.Tensor) -> tuple:
        """
        Args:
            z_s: [B, D] sentence embeddings
        Returns:
            (p, r_s): aspect distributions [B,K], reconstructions [B,D]
        """
        # Aspect distribution
        p   = F.softmax(self.W1(z_s), dim=-1)       # [B, K]

        # Reconstructed sentence embedding
        T_n = F.normalize(self.T, dim=-1)            # [K, D] normalised aspect matrix
        r_s = torch.matmul(p, T_n)                   # [B, D]

        return p, r_s

    def loss(self, z_s: torch.Tensor, r_s: torch.Tensor) -> torch.Tensor:
        """
        Contrastive max-margin loss (Eq 7 of He et al.):
          J = Σ_s [ max(0, 1 - cos(z_s, r_s) + cos(z_s, z_neg)) ]

        Negative samples are drawn randomly from the current minibatch.
        """
        B = z_s.shape[0]
        z_n = F.normalize(z_s, dim=-1)
        r_n = F.normalize(r_s, dim=-1)

        pos = (z_n * r_n).sum(dim=-1, keepdim=True)     # [B, 1]

        # Draw n_neg random negatives per sentence (from batch, with replacement)
        neg_idx = torch.randint(0, B, (B, self.n_neg), device=z_s.device)
        z_neg   = z_n[neg_idx]                          # [B, n_neg, D]
        neg     = (z_n.unsqueeze(1) * z_neg).sum(dim=-1) # [B, n_neg]

        # Max-margin
        margin  = F.relu(1.0 - pos + neg)               # [B, n_neg]
        return margin.mean()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_abae(embeddings: torch.Tensor,
               n_aspects: int = 5,
               epochs: int = 200,
               batch_size: int = 128,
               lr: float = 1e-3,
               n_neg: int = 20,
               device: str = 'cpu',
               seed: int = 42,
               patience: int = 10) -> ABAE:
    """
    Train ABAE on sentence embeddings.

    Args:
        embeddings: [N, D] sentence embeddings
        n_aspects:  number of aspect topics K
        epochs:     training epochs
        batch_size: minibatch size
        lr:         learning rate (Adam)
        n_neg:      negative samples per sentence
        device:     'cpu' or 'cuda'
        seed:       random seed

    Returns:
        trained ABAE model
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    D     = embeddings.shape[1]
    model = ABAE(embed_dim=D, n_aspects=n_aspects, n_neg=n_neg).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    dataset    = TensorDataset(embeddings.to(device))
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    n_batches  = len(loader)

    print(f'\nTraining ABAE: K={n_aspects}, epochs={epochs}, '
          f'batch={batch_size}, lr={lr}, n_neg={n_neg}')
    print(f'  N={len(embeddings)} sentences, D={D}')

    best_loss, patience_ct, best_state = float('inf'), 0, None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for (z_s,) in loader:
            p, r_s  = model(z_s)
            loss_v  = model.loss(z_s, r_s)
            opt.zero_grad()
            loss_v.backward()
            opt.step()
            total_loss += loss_v.item()

        avg = total_loss / n_batches
        if epoch % 20 == 0 or epoch == 1:
            print(f'  Epoch {epoch:>4}/{epochs}  loss={avg:.4f}')

        if patience > 0:
            if avg < best_loss - 1e-4:
                best_loss, patience_ct = avg, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_ct += 1
                if patience_ct >= patience:
                    print(f'  Early stop epoch {epoch}  best={best_loss:.4f}')
                    model.load_state_dict(best_state)
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Prediction and evaluation
# ---------------------------------------------------------------------------
def predict_aspects(model: ABAE,
                    embeddings: torch.Tensor,
                    batch_size: int = 256,
                    device: str = 'cpu') -> np.ndarray:
    """
    Predict aspect index for each sentence.
    Returns [N] int array of predicted aspect indices.
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            z_s  = embeddings[i:i+batch_size].to(device)
            p, _ = model(z_s)
            preds.append(p.argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds)


def hungarian_align(pred: np.ndarray,
                    true: np.ndarray,
                    K: int) -> np.ndarray:
    """Hungarian algorithm for optimal topic-to-label alignment."""
    cost = np.zeros((K, K))
    for k in range(K):
        for a in range(K):
            cost[k, a] = np.sum((pred == k) & (true == a))
    row, col = linear_sum_assignment(-cost)
    mapping  = dict(zip(row, col))
    return np.array([mapping.get(p, 0) for p in pred])


def evaluate_abae(model: ABAE,
                  embeddings: torch.Tensor,
                  true_labels: np.ndarray,
                  n_aspects: int,
                  device: str = 'cpu') -> dict:
    """
    Compute F1, accuracy, purity, NMI for ABAE predictions.
    Uses Hungarian alignment (same protocol as MAB-LDA evaluation).
    """
    pred_raw = predict_aspects(model, embeddings, device=device)
    valid    = true_labels >= 0
    pred_v   = pred_raw[valid]
    true_v   = true_labels[valid]

    pred_aligned = hungarian_align(pred_v, true_v, n_aspects)

    f1_mac = f1_score(true_v, pred_aligned, average='macro',    zero_division=0)
    f1_mic = f1_score(true_v, pred_aligned, average='micro',    zero_division=0)
    f1_wtd = f1_score(true_v, pred_aligned, average='weighted', zero_division=0)
    acc    = accuracy_score(true_v, pred_aligned)
    nmi    = normalized_mutual_info_score(true_v, pred_aligned)

    # Purity
    purity = sum(np.sum((pred_aligned == k) & (true_v == k))
                 for k in range(n_aspects)) / len(true_v)

    return {
        'f1_macro':    f1_mac,
        'f1_micro':    f1_mic,
        'f1_weighted': f1_wtd,
        'accuracy':    acc,
        'purity':      purity,
        'nmi':         nmi,
        'n_valid':     int(valid.sum()),
    }


# ---------------------------------------------------------------------------
# Dataset label extraction
# ---------------------------------------------------------------------------
ASPECT_COLS = {
    'semeval14_rest':   ['food', 'service', 'price', 'ambience', 'anecdotes'],
    'semeval14_laptop': ['performance', 'design', 'usability', 'price', 'support'],
    'semeval15_rest':   ['food', 'service', 'price', 'ambience', 'restaurant'],
    'semeval16_rest':   ['food', 'service', 'price', 'ambience', 'restaurant'],
    'mams':             ['food', 'service', 'price', 'ambience', 'restaurant',
                         'location', 'misc', 'drinks'],
}


def load_labels(data_path: str, dataset: str,
                model_name: str) -> tuple:
    """
    Load sentences and aspect labels from SemEval XML.
    Returns (texts, labels, aspect_cols) where labels[i]=-1 if no aspect.
    """
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
    loader = loaders[dataset]
    df, _, aspect_cols = loader(data_path, model_name)

    # Reconstruct text from input_ids via tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    texts = [tokenizer.decode(row, skip_special_tokens=True)
             for row in df['input_ids'].tolist()]

    # Majority-vote label per sentence
    labels = np.full(len(df), -1, dtype=int)
    aspect_matrix = df[aspect_cols].values       # [N, A]
    for i in range(len(df)):
        row = aspect_matrix[i]
        valid = [(row[a], a) for a in range(len(aspect_cols))
                 if not np.isnan(row[a])]
        if valid:
            labels[i] = max(valid, key=lambda x: x[0])[1]

    return texts, labels, aspect_cols


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description='ABAE: Unsupervised Aspect Extraction')
    p.add_argument('--dataset', type=str, required=True,
                   choices=list(ASPECT_COLS.keys()))
    p.add_argument('--train_path',           type=str, required=True)
    p.add_argument('--test_path',            type=str, default=None)
    p.add_argument('--model_name_or_path',   type=str,
                   default='bert-base-uncased')
    p.add_argument('--num_aspects',          type=int, default=5)
    p.add_argument('--epochs',               type=int, default=200)
    p.add_argument('--batch_size',           type=int, default=128)
    p.add_argument('--lr',                   type=float, default=1e-3)
    p.add_argument('--n_neg',                type=int, default=20)
    p.add_argument('--max_length',           type=int, default=128)
    p.add_argument('--eval_on_train',        action='store_true',
                   help='Evaluate on train split (default: test split)')
    p.add_argument('--output_csv',           type=str,
                   default='./results/abae_results.csv')
    p.add_argument('--patience',             type=int, default=10,
                   help='Early stopping patience in epochs (0=disabled)')
    p.add_argument('--seed',                 type=int, default=42)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print(time.strftime('%a, %d %b %Y %H:%M:%S'))

    # -----------------------------------------------------------------------
    # 1. Load data and labels
    # -----------------------------------------------------------------------
    print(f'\nLoading train data: {args.train_path}')
    train_texts, train_labels, aspect_cols = load_labels(
        args.train_path, args.dataset, args.model_name_or_path)

    if args.test_path and not args.eval_on_train:
        print(f'Loading test data:  {args.test_path}')
        eval_texts, eval_labels, _ = load_labels(
            args.test_path, args.dataset, args.model_name_or_path)
    else:
        eval_texts, eval_labels = train_texts, train_labels

    # Combine train+test for encoding (ABAE is unsupervised — uses all text)
    all_texts = train_texts
    if args.test_path:
        test_texts, _, _ = load_labels(
            args.test_path, args.dataset, args.model_name_or_path)
        all_texts = train_texts + test_texts

    print(f'Train docs: {len(train_texts)}'
          f'  Eval docs: {len(eval_texts)}'
          f'  Total: {len(all_texts)}')

    # -----------------------------------------------------------------------
    # 2. Encode sentences with BERT
    # -----------------------------------------------------------------------
    all_embeddings = encode_sentences(
        all_texts, args.model_name_or_path,
        max_length=args.max_length, device=device)

    # Eval split embeddings
    if args.test_path and not args.eval_on_train:
        eval_embeddings = all_embeddings[len(train_texts):]
    else:
        eval_embeddings = all_embeddings[:len(train_texts)]

    # -----------------------------------------------------------------------
    # 3. Train ABAE
    # -----------------------------------------------------------------------
    model = train_abae(
        all_embeddings,
        n_aspects  = args.num_aspects,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        n_neg      = args.n_neg,
        device     = device,
        seed       = args.seed,
        patience   = args.patience,
    )

    # -----------------------------------------------------------------------
    # 4. Evaluate
    # -----------------------------------------------------------------------
    print('\nEvaluating...')
    metrics = evaluate_abae(
        model, eval_embeddings, eval_labels,
        args.num_aspects, device=device)

    print()
    print(f'=== ABAE Results ({args.dataset}) ===')
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f'  {k:<18}: {v:.4f}')
        else:
            print(f'  {k:<18}: {v}')

    # -----------------------------------------------------------------------
    # 5. Save results
    # -----------------------------------------------------------------------
    import pandas as pd
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    row = {
        'method':       'ABAE (He 2017)',
        'dataset':      args.dataset,
        'model':        args.model_name_or_path,
        'K':            args.num_aspects,
        'epochs':       args.epochs,
        **{k: round(v, 4) if isinstance(v, float) else v
           for k, v in metrics.items()},
    }
    df_row = pd.DataFrame([row])
    if Path(args.output_csv).exists():
        df_row.to_csv(args.output_csv, mode='a', header=False, index=False)
    else:
        df_row.to_csv(args.output_csv, index=False)
    print(f'\nSaved to {args.output_csv}')
    print(time.strftime('%a, %d %b %Y %H:%M:%S'))


if __name__ == '__main__':
    main()
