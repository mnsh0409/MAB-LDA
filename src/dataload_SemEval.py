"""
dataload_SemEval.py  — revised for ICDM submission
====================================================
Adds loaders for all required benchmark datasets:

  EXISTING (unchanged API):
    get_semeval_data()       — SemEval-2014 Restaurant (XML, 5 aspect cats)

  NEW — SemEval family:
    get_semeval15_data()     — SemEval-2015 Restaurant (XML, E#A format)
    get_semeval16_data()     — SemEval-2016 Restaurant (XML, E#A format, alias of 15 parser)
    get_semeval14_laptop()   — SemEval-2014 Laptop     (XML, 5 aspect cats)

  NEW — Hard multi-aspect:
    get_mams_data()          — MAMS ACSA (JSON/CSV, 8 aspect cats)

  NEW — Topic model coherence benchmark:
    get_20newsgroups()       — 20 Newsgroups (scikit-learn, K=20 ground truth)

  SHARED HELPERS (unchanged):
    tokenize_data(), lemmatize_text_lower(), LLRDataset, split_aspect(), ...

Dataset paths (edit DATA_ROOT or pass file_path explicitly):

  SemEval-2014 Restaurant Train : data/SemEval14/Restaurants_Train_v2.xml
  SemEval-2014 Restaurant Test  : data/SemEval14/Restaurants_Test_Data_phaseB.xml
  SemEval-2014 Laptop Train     : data/SemEval14/Laptop_Train_v2.xml
  SemEval-2014 Laptop Test      : data/SemEval14/Laptops_Test_Data_phaseB.xml
  SemEval-2015 Restaurant Train : data/SemEval15/ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml
  SemEval-2015 Restaurant Test  : data/SemEval15/ABSA15_RestaurantsTest.xml
  SemEval-2016 Restaurant Train : data/SemEval16/ABSA16_Restaurants_Train_SB1_v2.xml
  SemEval-2016 Restaurant Test  : data/SemEval16/EN_REST_SB1_TEST.xml.gold
  MAMS ACSA Train               : data/MAMS/train.xml   (or train.csv)
  MAMS ACSA Test                : data/MAMS/test.xml    (or test.csv)
  20 Newsgroups                 : auto-downloaded by scikit-learn

Official download links:
  SemEval-2014: https://alt.qcri.org/semeval2014/task4/
  SemEval-2015: https://alt.qcri.org/semeval2015/task12/
  SemEval-2016: https://alt.qcri.org/semeval2016/task5/
  MAMS:         https://github.com/siat-nlp/MAMS-for-ABSA
  20NG:         sklearn.datasets.fetch_20newsgroups (auto)
"""

import re
import os
import xml.etree.ElementTree as ET
import logging
import math
import ast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from transformers import (
    AutoTokenizer, AdamW,
    T5Config, T5PreTrainedModel, T5Tokenizer, T5TokenizerFast, T5Model, T5EncoderModel,
)
from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, Subset, RandomSampler, SubsetRandomSampler
from sklearn.metrics import classification_report, accuracy_score, f1_score

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Shared NLP helpers (unchanged from original)
# ---------------------------------------------------------------------------
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
stop_words = set(stopwords.words('english'))
stop_words.update([
    'dass', 'nicht', 'sont', 'come', 'from', 'with', 'that', 'this', 'they',
    'have', 'were', 'food', 'place', 'restaurant', 'good', 'great', 'one',
    'would', 'like', 'get', 'always', 'really', 'also', 'even', 'much',
    'well', 'time', 'went', 'make', 'over',
])


def lemmatize_text_lower(text):
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer  = nltk.stem.WordNetLemmatizer()
    try:
        text = text.lower()
        return TreebankWordDetokenizer().detokenize(
            [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
        )
    except Exception:
        return ''


def tokenize_data(data, MODEL_PATH):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokens = tokenizer(list(data), padding='max_length', max_length=512, truncation=True)
    return tokens, tokenizer


# ---------------------------------------------------------------------------
# Polarity mapping helpers
# ---------------------------------------------------------------------------
def _polarity_str_to_int(polarity: str) -> int:
    """Shared polarity converter used by all SemEval parsers."""
    if polarity == 'positive':
        return 1
    elif polarity == 'negative':
        return 0
    else:
        return 2   # neutral or conflict


def _build_sampler_tokens(tokenizer, words: str, n: int = 9) -> list:
    """Build the fixed-length anchor token list for the Gibbs sampler."""
    return tokenizer(words, padding='max_length', max_length=n).input_ids


def _finalize_df(data: pd.DataFrame, aspect_cols: list,
                 model_path: str, anchor_str: str) -> tuple:
    """
    Shared post-processing:
      1. lemmatize ReviewText
      2. tokenize
      3. compute AspectAVG
      4. build sampler_tokens
    Returns (data, sampler_tokens, aspect_cols)
    """
    data['ReviewTitle'] = ''
    data['ReviewText']  = data['ReviewText'].apply(lemmatize_text_lower)
    tokens, tokenizer   = tokenize_data(
        ' ' + data['ReviewText'].fillna('').values, model_path
    )
    data['input_ids']      = list(tokens.input_ids)
    data['attention_mask'] = list(tokens.attention_mask)
    data['AspectAVG']      = (data[aspect_cols]
                               .replace(2, np.nan)
                               .mean(axis=1, skipna=True, numeric_only=True))
    sampler_tokens = _build_sampler_tokens(tokenizer, anchor_str)
    return data, sampler_tokens, aspect_cols


# ---------------------------------------------------------------------------
# LLRDataset (unchanged from original)
# ---------------------------------------------------------------------------
class LLRDataset(Dataset):
    def __init__(self, data, llr_words):
        self.input_ids     = torch.LongTensor(list(data.input_ids))
        self.attention_mask = torch.LongTensor(list(data.attention_mask))
        self.llr_ids = (torch.LongTensor(list(data.llr_ids))
                        if 'llr_ids' in data.columns
                        else torch.zeros(data.shape[0], 1).long())
        self.labels   = (torch.LongTensor(list(data.labels))
                         if 'labels' in data.columns else None)
        self.llr_words = llr_words
        if 'index' in data.columns:
            self.index = torch.LongTensor(list(data.index))

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        if self.labels is None:
            outputs = (self.input_ids[idx], self.attention_mask[idx], self.llr_ids[idx])
        else:
            outputs = (self.input_ids[idx], self.attention_mask[idx],
                       self.llr_ids[idx], self.labels[idx])
        return idx, outputs


# ===========================================================================
# DATASET 1 (EXISTING) — SemEval-2014 Restaurant
# ===========================================================================
SEMEVAL14_REST_ASPECTS = ['food', 'service', 'price', 'ambience', 'anecdotes/miscellaneous']
SEMEVAL14_REST_ANCHOR  = ' food service price ambience in.'


def get_semeval_data(FILE_PATH: str, MODEL_PATH: str) -> tuple:
    """
    Parses SemEval-2014 Task 4 Restaurant XML.
    UNCHANGED API — drop-in for existing callers.

    Returns: (df, sampler_tokens, aspect_cols)
    """
    tree = ET.parse(FILE_PATH)
    root = tree.getroot()
    aspect_cols = SEMEVAL14_REST_ASPECTS
    data_list   = []

    for sentence in root.findall('sentence'):
        text_node = sentence.find('text')
        if text_node is None or text_node.text is None:
            continue
        aspect_dict = {col: np.nan for col in aspect_cols}
        cats = sentence.find('aspectCategories')
        if cats is not None:
            for cat in cats.findall('aspectCategory'):
                name = cat.get('category')
                if name in aspect_dict:
                    aspect_dict[name] = _polarity_str_to_int(cat.get('polarity', ''))
        row = {'ReviewText': text_node.text}
        row.update(aspect_dict)
        data_list.append(row)

    data = pd.DataFrame(data_list)
    return _finalize_df(data, aspect_cols, MODEL_PATH, SEMEVAL14_REST_ANCHOR)


# ===========================================================================
# DATASET 2 (NEW) — SemEval-2014 Laptop
# ===========================================================================
# Laptop XML uses <aspectTerms> (term-level) not <aspectCategories>.
# We map them to 5 coarse categories matching the laptop annotation guide.
SEMEVAL14_LAPTOP_ASPECTS = ['performance', 'design', 'usability', 'price', 'support']
SEMEVAL14_LAPTOP_ANCHOR  = ' performance design usability price support in.'

# Surface-form → coarse category mapping (covers the most frequent terms)
_LAPTOP_TERM_MAP = {
    'performance': ['speed', 'performance', 'battery', 'processor', 'cpu', 'ram',
                    'memory', 'graphics', 'display', 'screen', 'resolution'],
    'design':      ['design', 'build', 'size', 'weight', 'keyboard', 'trackpad',
                    'touchpad', 'port', 'speaker', 'webcam', 'camera'],
    'usability':   ['software', 'os', 'windows', 'linux', 'boot', 'driver',
                    'fan', 'heat', 'noise', 'wifi', 'bluetooth'],
    'price':       ['price', 'cost', 'value', 'worth', 'money', 'deal', 'expensive',
                    'cheap', 'affordable'],
    'support':     ['support', 'warranty', 'service', 'customer', 'repair', 'replace'],
}


def _map_laptop_term(term: str) -> str:
    """Map a raw aspect term string to one of the 5 coarse categories."""
    term_lower = term.lower()
    for cat, keywords in _LAPTOP_TERM_MAP.items():
        if any(k in term_lower for k in keywords):
            return cat
    return 'usability'   # fallback


def get_semeval14_laptop(FILE_PATH: str, MODEL_PATH: str) -> tuple:
    """
    Parses SemEval-2014 Task 4 Laptop XML (aspect-term level annotations).
    Maps raw terms → 5 coarse categories to match the model's fixed-K structure.

    Data file: Laptop_Train_v2.xml / Laptops_Test_Data_phaseB.xml
    Download:  https://alt.qcri.org/semeval2014/task4/

    Returns: (df, sampler_tokens, aspect_cols)
    """
    tree = ET.parse(FILE_PATH)
    root = tree.getroot()
    aspect_cols = SEMEVAL14_LAPTOP_ASPECTS
    data_list   = []

    for sentence in root.findall('sentence'):
        text_node = sentence.find('text')
        if text_node is None or text_node.text is None:
            continue
        aspect_dict = {col: np.nan for col in aspect_cols}
        terms = sentence.find('aspectTerms')
        if terms is not None:
            for term in terms.findall('aspectTerm'):
                raw_term = term.get('term', '')
                polarity = term.get('polarity', '')
                cat      = _map_laptop_term(raw_term)
                # If the sentence has conflicting polarities for same cat, keep first
                if np.isnan(float(aspect_dict[cat])) if not isinstance(aspect_dict[cat], float) else np.isnan(aspect_dict[cat]):
                    aspect_dict[cat] = _polarity_str_to_int(polarity)
        row = {'ReviewText': text_node.text}
        row.update(aspect_dict)
        data_list.append(row)

    data = pd.DataFrame(data_list)
    return _finalize_df(data, aspect_cols, MODEL_PATH, SEMEVAL14_LAPTOP_ANCHOR)


# ===========================================================================
# DATASET 3 (NEW) — SemEval-2015 Restaurant  (E#A format)
# ===========================================================================
# SemEval-2015/2016 use Entity#Attribute pairs, e.g. FOOD#QUALITY.
# We group these into 5 entity-level buckets matching the 2014 scheme
# so the same Gibbs anchor setup works unchanged.
SEMEVAL15_REST_ASPECTS = ['food', 'service', 'price', 'ambience', 'restaurant']
SEMEVAL15_REST_ANCHOR  = ' food service price ambience restaurant in.'

_SE15_ENTITY_MAP = {
    'FOOD':        'food',
    'DRINKS':      'food',        # beverages = food aspect for our purposes
    'SERVICE':     'service',
    'PRICES':      'price',
    'AMBIENCE':    'ambience',
    'LOCATION':    'ambience',
    'RESTAURANT':  'restaurant',
}


def _parse_semeval15_xml(FILE_PATH: str, aspect_cols: list,
                          entity_map: dict) -> pd.DataFrame:
    """
    Generic parser for SemEval-2015/2016 E#A XML format.
    Handles both <Review>/<sentences>/<sentence> (2015) and
    flat <sentences>/<sentence> (2016 variants).
    """
    tree = ET.parse(FILE_PATH)
    root = tree.getroot()
    data_list = []

    # 2015 wraps sentences inside <Review> elements; 2016 may be flat
    sentences = root.findall('.//sentence')

    for sentence in sentences:
        text_node = sentence.find('text')
        if text_node is None or text_node.text is None:
            continue
        aspect_dict = {col: np.nan for col in aspect_cols}
        opinions = sentence.find('Opinions')
        if opinions is not None:
            for opinion in opinions.findall('Opinion'):
                category = opinion.get('category', '')   # e.g. "FOOD#QUALITY"
                polarity = opinion.get('polarity', '')
                entity   = category.split('#')[0] if '#' in category else category
                col      = entity_map.get(entity)
                if col and col in aspect_dict:
                    if isinstance(aspect_dict[col], float) and np.isnan(aspect_dict[col]):
                        aspect_dict[col] = _polarity_str_to_int(polarity)
        row = {'ReviewText': text_node.text}
        row.update(aspect_dict)
        data_list.append(row)

    return pd.DataFrame(data_list)


def get_semeval15_data(FILE_PATH: str, MODEL_PATH: str) -> tuple:
    """
    Parses SemEval-2015 Task 12 Restaurant XML (E#A format).

    Data files: ABSA15_Restaurants_Train_Final.xml / ABSA15_RestaurantsTest.xml
    Download:   https://alt.qcri.org/semeval2015/task12/

    Returns: (df, sampler_tokens, aspect_cols)
    """
    aspect_cols = SEMEVAL15_REST_ASPECTS
    data        = _parse_semeval15_xml(FILE_PATH, aspect_cols, _SE15_ENTITY_MAP)
    return _finalize_df(data, aspect_cols, MODEL_PATH, SEMEVAL15_REST_ANCHOR)


# ===========================================================================
# DATASET 4 (NEW) — SemEval-2016 Restaurant (E#A format, same parser as 2015)
# ===========================================================================
SEMEVAL16_REST_ASPECTS = SEMEVAL15_REST_ASPECTS   # same entity scheme
SEMEVAL16_REST_ANCHOR  = SEMEVAL15_REST_ANCHOR


def get_semeval16_data(FILE_PATH: str, MODEL_PATH: str) -> tuple:
    """
    Parses SemEval-2016 Task 5 Restaurant XML (EN, SB1).
    Uses the same E#A parser as SemEval-2015 — schema is identical.

    Data files: ABSA16_Restaurants_Train_SB1_v2.xml / EN_REST_SB1_TEST.xml.gold
    Download:   https://alt.qcri.org/semeval2016/task5/

    Returns: (df, sampler_tokens, aspect_cols)
    """
    aspect_cols = SEMEVAL16_REST_ASPECTS
    data        = _parse_semeval15_xml(FILE_PATH, aspect_cols, _SE15_ENTITY_MAP)
    return _finalize_df(data, aspect_cols, MODEL_PATH, SEMEVAL16_REST_ANCHOR)


# ===========================================================================
# DATASET 5 (NEW) — MAMS (Multi-Aspect Multi-Sentiment)
# ===========================================================================
# MAMS ACSA uses 8 aspect categories — a perfect match for your n_topics=8.
MAMS_ASPECTS = ['food', 'service', 'price', 'ambience',
                'location', 'drinks', 'restaurant', 'miscellaneous']
MAMS_ANCHOR  = ' food service price ambience location drinks restaurant miscellaneous in.'

# MAMS XML <category> strings → our column names
_MAMS_CAT_MAP = {
    'food':          'food',
    'service':       'service',
    'price':         'price',
    'ambience':      'ambience',
    'location':      'location',
    'drinks':        'drinks',
    'restaurant':    'restaurant',
    'anecdotes/miscellaneous': 'miscellaneous',
    'miscellaneous': 'miscellaneous',
}


def get_mams_data(FILE_PATH: str, MODEL_PATH: str) -> tuple:
    """
    Parses MAMS ACSA dataset.
    Supports both XML and CSV formats (auto-detected by extension).

    MAMS is the hardest test for your exclusivity reward because every
    sentence contains ≥2 aspects with DIFFERENT polarities — the model
    cannot take a shortcut by assigning the whole sentence one polarity.

    Data files: train.xml, val.xml, test.xml  (or .csv variants)
    Download:   https://github.com/siat-nlp/MAMS-for-ABSA
    Paper:      Jiang et al., EMNLP 2019

    Returns: (df, sampler_tokens, aspect_cols)
    """
    ext         = os.path.splitext(FILE_PATH)[1].lower()
    aspect_cols = MAMS_ASPECTS

    if ext in ('.xml',):
        data = _parse_mams_xml(FILE_PATH, aspect_cols)
    elif ext in ('.csv', '.tsv'):
        data = _parse_mams_csv(FILE_PATH, aspect_cols)
    else:
        # Try XML first, fall back to CSV
        try:
            data = _parse_mams_xml(FILE_PATH, aspect_cols)
        except ET.ParseError:
            data = _parse_mams_csv(FILE_PATH, aspect_cols)

    return _finalize_df(data, aspect_cols, MODEL_PATH, MAMS_ANCHOR)


def _parse_mams_xml(FILE_PATH: str, aspect_cols: list) -> pd.DataFrame:
    """
    MAMS XML format mirrors SemEval-2014:
      <sentence id="...">
        <text>...</text>
        <aspectCategories>
          <aspectCategory category="food" polarity="positive"/>
          ...
        </aspectCategories>
      </sentence>
    """
    tree = ET.parse(FILE_PATH)
    root = tree.getroot()
    data_list = []

    for sentence in root.findall('.//sentence'):
        text_node = sentence.find('text')
        if text_node is None or text_node.text is None:
            continue
        aspect_dict = {col: np.nan for col in aspect_cols}
        cats = sentence.find('aspectCategories')
        if cats is not None:
            for cat in cats.findall('aspectCategory'):
                raw  = cat.get('category', '').lower().strip()
                col  = _MAMS_CAT_MAP.get(raw)
                if col:
                    pol = _polarity_str_to_int(cat.get('polarity', ''))
                    if isinstance(aspect_dict[col], float) and np.isnan(aspect_dict[col]):
                        aspect_dict[col] = pol
        row = {'ReviewText': text_node.text}
        row.update(aspect_dict)
        data_list.append(row)

    return pd.DataFrame(data_list)


def _parse_mams_csv(FILE_PATH: str, aspect_cols: list) -> pd.DataFrame:
    """
    MAMS CSV format (preprocessed variant from the GitHub repo):
      text, food, service, price, ambience, location, drinks, restaurant, miscellaneous
    Polarities encoded as: positive=1, negative=0, neutral=2, none=NaN
    """
    raw = pd.read_csv(FILE_PATH)
    data_list = []

    for _, row_raw in raw.iterrows():
        text = str(row_raw.get('text', row_raw.iloc[0]))
        aspect_dict = {}
        for col in aspect_cols:
            val = row_raw.get(col, None)
            if val is None or (isinstance(val, float) and np.isnan(val)) or str(val).lower() == 'none':
                aspect_dict[col] = np.nan
            elif str(val).lower() == 'positive':
                aspect_dict[col] = 1
            elif str(val).lower() == 'negative':
                aspect_dict[col] = 0
            else:
                try:
                    aspect_dict[col] = int(val)
                except (ValueError, TypeError):
                    aspect_dict[col] = np.nan
        row = {'ReviewText': text}
        row.update(aspect_dict)
        data_list.append(row)

    return pd.DataFrame(data_list)


# ===========================================================================
# DATASET 6 (NEW) — 20 Newsgroups (topic coherence benchmark)
# ===========================================================================
# 20 Newsgroups has K=20 ground-truth topics, making it the standard
# benchmark for reporting NPMI coherence and topic diversity alongside
# log-likelihood convergence curves.
NG20_ASPECTS = [f'topic_{i}' for i in range(20)]   # 20 dummy label columns
NG20_ANCHOR  = ' computer politics religion science sports car in.'


def get_20newsgroups(MODEL_PATH: str,
                     subset: str = 'all',
                     categories=None,
                     max_docs: int = 5000,
                     remove: tuple = ('headers', 'footers', 'quotes')) -> tuple:
    """
    Loads the 20 Newsgroups dataset via scikit-learn (auto-downloaded).
    No file path needed — sklearn fetches and caches it automatically.

    Args:
        MODEL_PATH:  HuggingFace model name for tokenization
        subset:      'train', 'test', or 'all'
        categories:  None = all 20; or list of category names to subset
        max_docs:    cap on number of documents (default 5000 for speed)
        remove:      metadata to strip from posts (headers/footers/quotes
                     are noise that inflates coherence scores artificially)

    Returns: (df, sampler_tokens, ['target_name'])

    Note on aspect_cols:
        20NG has no aspect labels — we expose the ground-truth category
        name as a single 'target_name' column for coherence evaluation.
        Pass aspect_cols=['target_name'] to any downstream evaluator.
    """
    try:
        from sklearn.datasets import fetch_20newsgroups
    except ImportError:
        raise ImportError("scikit-learn is required: pip install scikit-learn")

    print(f"Loading 20 Newsgroups (subset='{subset}', max_docs={max_docs})...")
    ng = fetch_20newsgroups(subset=subset, categories=categories,
                            remove=remove, shuffle=True, random_state=42)

    texts   = ng.data[:max_docs]
    targets = ng.target[:max_docs]
    names   = np.array(ng.target_names)

    data = pd.DataFrame({
        'ReviewText':  texts,
        'target':      targets,
        'target_name': names[targets],
    })

    # No aspect polarities — fill dummy column so _finalize_df doesn't break
    data['target_label'] = data['target']   # integer 0-19

    data['ReviewTitle'] = ''
    data['ReviewText']  = data['ReviewText'].apply(lemmatize_text_lower)
    tokens, tokenizer   = tokenize_data(
        ' ' + data['ReviewText'].fillna('').values, MODEL_PATH
    )
    data['input_ids']      = list(tokens.input_ids)
    data['attention_mask'] = list(tokens.attention_mask)
    # No meaningful AspectAVG — set to neutral 0.5
    data['AspectAVG'] = 0.5

    sampler_tokens = _build_sampler_tokens(tokenizer, NG20_ANCHOR)
    aspect_cols    = ['target_label']   # used only for AspectAVG fallback

    print(f"20 Newsgroups loaded: {len(data)} docs, "
          f"{len(ng.target_names)} categories")
    return data, sampler_tokens, aspect_cols


# ===========================================================================
# Aviation & original data helpers (unchanged from original dataload_SemEval)
# ===========================================================================
def split_aspect(data):
    temp = np.full((8, data.shape[0]), 2, int)
    for idx in range(data.shape[0]):
        aspect = data[idx]
        for i, asp in enumerate(
            ['Legroom', 'Seat', 'Entertainment', 'Customer',
             'Value', 'Cleanliness', 'Check-in', 'Food']
        ):
            for sub_asp in aspect:
                if asp in sub_asp:
                    pol = int(sub_asp[-1])
                    temp[i, idx] = 1 if pol > 3 else 0
                    break
    return temp


def split_aspectRAW(data):
    temp = np.full((8, data.shape[0]), np.nan, float)
    for idx in range(data.shape[0]):
        aspect = data[idx]
        for i, asp in enumerate(
            ['Legroom', 'Seat', 'Entertainment', 'Customer',
             'Value', 'Cleanliness', 'Check-in', 'Food']
        ):
            for sub_asp in aspect:
                if asp in sub_asp:
                    pol = int(sub_asp[-1])
                    temp[i, idx] = pol
                    break
    return temp


def get_data(FILE_PATH, MODEL_PATH, COL_NAMES):
    """Original aviation data loader — unchanged."""
    ASPECT_NAMES = ['LEG', 'SIT', 'ENT', 'CUS', 'VOM', 'CLE', 'CKI', 'FNB']
    raw_data = pd.read_csv(FILE_PATH, sep='\t', header=None, names=COL_NAMES)
    data = raw_data
    data['Rating'] = list(map(lambda x: 1 if x > 3 else 0, data['Rating']))
    data['Year']   = [y[-4:] for y in data['ReviewDate']]
    data = country_region(data)
    data = pandemic(data)
    data.Aspects = data.Aspects.str.split('|').values

    aspects_splitted = split_aspect(data.Aspects.values)
    for i in range(len(ASPECT_NAMES)):
        data[ASPECT_NAMES[i]] = aspects_splitted[i, :]

    data['AspectAVG'] = (data[['LEG', 'SIT', 'ENT', 'CUS', 'VOM', 'CLE', 'CKI', 'FNB']]
                         .replace(2, np.nan)
                         .mean(axis=1, skipna=True, numeric_only=True))
    data['ReviewTitle'] = data.ReviewTitle.apply(lemmatize_text_lower)
    data['ReviewText']  = data.ReviewText.apply(lemmatize_text_lower)

    tokens, tokenizer = tokenize_data(
        data.ReviewTitle.map(str).values + '. ' + data.ReviewText.values, MODEL_PATH
    )
    data['input_ids']      = list(tokens.input_ids)
    data['attention_mask'] = list(tokens.attention_mask)
    tokens = tokenizer(
        ' airline trip Trip staff in.', padding='max_length', max_length=9
    ).input_ids
    return data, tokens


def country_region(df):
    conditions = [
        (df['AirlineName'] == 'Azul'),
        (df['AirlineName'] == 'Emirates'),
        (df['AirlineName'] == 'ANA (All Nippon Airways)') | (df['AirlineName'] == 'Japan Airlines (JAL)'),
        (df['AirlineName'] == 'Air New Zealand'),
        (df['AirlineName'] == 'Qatar Airways'),
        (df['AirlineName'] == 'Singapore Airlines'),
        (df['AirlineName'] == 'EVA Air'),
        (df['AirlineName'] == 'Jet2.com'),
        (df['AirlineName'] == 'Southwest Airlines'),
    ]
    values = ['Brazil', 'Dubai', 'Japan', 'New Zealand', 'Qatar',
              'Singapore', 'Taiwan', 'UK', 'US']
    df['Country'] = np.select(conditions, values)

    conditions = [
        (df['Country'] == 'Japan') | (df['Country'] == 'Singapore') | (df['Country'] == 'Taiwan'),
        (df['Country'] == 'Dubai') | (df['Country'] == 'Qatar'),
        (df['Country'] == 'UK'),
        (df['Country'] == 'US') | (df['Country'] == 'Brazil'),
        (df['Country'] == 'New Zealand'),
    ]
    values = ['East Asia', 'West Asia', 'Europe', 'America', 'Oceania']
    df['Region'] = np.select(conditions, values)
    return df


def pandemic(df):
    conditions = [
        (df['Year'].map(int) < 2020),
        (df['Year'].map(int) >= 2020) & (df['Year'].map(int) < 2023),
        (df['Year'].map(int) == 2023),
    ]
    df['pandemic'] = np.select(conditions, [0, 1, 1])
    return df


def word_class_freq(data, aspect_name, aspect_class=3):
    temp = np.zeros((33000, aspect_class), int)
    ids    = data.input_ids.values
    labels = data[aspect_name].values
    for sub_ids, sub_lb in zip(ids, labels):
        set_ids = set(sub_ids)
        for ids in set_ids:
            temp[ids, sub_lb] += 1
    return temp


def calculate_llr(temp_df, labels, data):
    import math as _math
    N = data.shape[0]
    total_scores = []
    for i in temp_df.index.values:
        llr_scores = []
        for class_ in [0, 1, 2]:
            num_class_doc = np.sum(labels == class_)
            n11 = temp_df.loc[i, class_]
            n10 = num_class_doc - n11
            n01 = temp_df.loc[i, 'total'] - n11
            n00 = N - n11 - n10 - n01
            pt  = (1e-10 + n11 + n01) / N
            p1  = n11 / (1e-10 + n11 + n10)
            p2  = n01 / (1e-10 + n01 + n00)
            try:
                e1 = n11 * (_math.log(pt) - _math.log(p1))
            except Exception:
                e1 = 0
            try:
                e2 = n10 * (_math.log(1 - pt) - _math.log(1 - p1))
            except Exception:
                e2 = 0
            try:
                e3 = n01 * (_math.log(pt) - _math.log(p2))
            except Exception:
                e3 = 0
            try:
                e4 = n00 * (_math.log(1 - pt) - _math.log(1 - p2))
            except Exception:
                e4 = 0
            llr_score = -2 * (e1 + e2 + e3 + e4)
            if n11 < n01:
                llr_score = 0
            llr_scores.append(llr_score)
        total_scores.append(llr_scores)
    llr_df = pd.DataFrame(
        np.array(total_scores),
        index=temp_df.index,
        columns=temp_df.columns.values[:-1]
    )
    return llr_df


def generate_llr_score(data, aspect):
    temp    = word_class_freq(data, aspect)
    temp_df = pd.DataFrame(temp)
    temp_df['total'] = np.sum(temp, -1)
    temp_df = temp_df[temp_df['total'] != 0]
    temp_df = temp_df.drop(0, axis=0)
    return calculate_llr(temp_df, data[aspect].values, data)
