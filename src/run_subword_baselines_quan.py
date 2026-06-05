import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import BertTokenizer
from eval_metrics import hungarian_align, DATASET_ASPECTS
from sklearn.metrics import f1_score, normalized_mutual_info_score
import argparse

# Import your own robust data loaders
from dataload_SemEval import (
    get_semeval_data, get_semeval14_laptop,
    get_semeval15_data, get_semeval16_data, get_mams_data
)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def load_and_tokenize_via_df(filepath, dataset_name):
    # Map dataset name to your specific loader function
    loaders = {
        'semeval14_rest':   get_semeval_data,
        'semeval14_laptop': get_semeval14_laptop,
        'semeval15_rest':   get_semeval15_data,
        'semeval16_rest':   get_semeval16_data,
        'mams':             get_mams_data,
    }
    
    loader = loaders.get(dataset_name)
    if not loader:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    # Load data using your pipeline
    df, _, _ = loader(filepath, 'bert-base-uncased')
    aspect_cols = DATASET_ASPECTS.get(dataset_name, [])
    
    # Replicate your eval_metrics.py logic to extract the dominant aspect
    aspect_matrix_full = df[aspect_cols].values 
    D = len(df)
    labels = np.full(D, -1, dtype=int)
    
    for d in range(D):
        for a in range(len(aspect_cols)):
            val = aspect_matrix_full[d, a]
            if not (isinstance(val, float) and np.isnan(val)):
                labels[d] = a 
                break
                
    # Convert the pre-computed integer IDs directly back into subword strings
    subword_texts = []
    for ids in df['input_ids'].tolist():
        # Convert integers back to tokens (e.g., '[CLS]', 'food', '##s')
        tokens = tokenizer.convert_ids_to_tokens(ids)
        
        # Filter out [PAD] tokens so they don't artificially skew the sklearn counts
        tokens = [tok for tok in tokens if tok != '[PAD]']
        
        subword_texts.append(" ".join(tokens))
        
    return subword_texts, labels

def compute_metrics(y_true, y_pred, num_topics):
    valid = y_true != -1
    y_t = y_true[valid]
    y_p = y_pred[valid]
    
    if len(y_t) == 0:
        print("WARNING: No valid labels found!")
        return {'f1_macro': 0.0, 'nmi': 0.0}

    mapping = hungarian_align(y_p, y_t, num_topics, num_topics)
    pred_aligned = np.array([mapping[t] for t in y_p])
    
    f1 = f1_score(y_t, pred_aligned, average='macro', zero_division=0)
    nmi = normalized_mutual_info_score(y_t, y_p)
    return {'f1_macro': f1, 'nmi': nmi}

def run_subword_collapse_test(dataset_path, dataset_name, num_topics):
    print(f"\n{'='*50}\nRunning Subword Collapse Baselines on {dataset_name}\n{'='*50}")
    
    X_text, y_true = load_and_tokenize_via_df(dataset_path, dataset_name)
    
    print(f"Loaded {len(X_text)} documents. Valid labels: {(y_true != -1).sum()}")
    
    count_vec = CountVectorizer(lowercase=False, token_pattern=r"(?u)\b\S+\b") 
    X_counts = count_vec.fit_transform(X_text)
    
    tfidf_vec = TfidfVectorizer(lowercase=False, token_pattern=r"(?u)\b\S+\b")
    X_tfidf = tfidf_vec.fit_transform(X_text)

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_doc_topic = lda.fit_transform(X_counts)
    lda_preds = np.argmax(lda_doc_topic, axis=1)
    
    nmf = NMF(n_components=num_topics, random_state=42, init='nndsvd')
    nmf_doc_topic = nmf.fit_transform(X_tfidf)
    nmf_preds = np.argmax(nmf_doc_topic, axis=1)

    lda_metrics = compute_metrics(y_true, lda_preds, num_topics)
    print(f"\n[RESULTS] Standard LDA (BERT Subwords):")
    print(f"F1 Macro: {lda_metrics['f1_macro']:.3f} | NMI: {lda_metrics['nmi']:.3f}")

    nmf_metrics = compute_metrics(y_true, nmf_preds, num_topics)
    print(f"\n[RESULTS] NMF (BERT Subwords):")
    print(f"F1 Macro: {nmf_metrics['f1_macro']:.3f} | NMI: {nmf_metrics['nmi']:.3f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--test_path', required=True)
    parser.add_argument('--num_aspects', type=int, required=True)
    args = parser.parse_args()
    
    run_subword_collapse_test(args.test_path, args.dataset, args.num_aspects)
