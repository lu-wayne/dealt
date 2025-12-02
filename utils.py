import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter
import numpy as np
import torch
import pandas as pd
import json
from tqdm import tqdm

# Load Sentence-BERT model once
SBERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

def load_data(file_path):
    """ Loads data from a CSV file """
    
    return pd.read_csv(file_path)

def compute_embeddings(texts):
    """ Computes Sentence-BERT embeddings for a list of texts """

    if isinstance(texts, str):
        texts = [texts]

    # Remove None values or empty strings
    texts = [text for text in texts if text]
    if not texts:
        return np.array([]) 

    embeddings = SBERT_MODEL.encode(texts, convert_to_numpy=True)
    return normalize(embeddings) # Normalize embeddings


def compute_semantic_dissimilarity(emb1, emb2):
    """ Computes cosine distance between two embeddings """

    if emb1 is None or emb2 is None or emb1.shape[0] == 0 or emb2.shape[0] == 0:
         return 0.0 # Handle empty inputs

    # Compute cosine similarity (1 - cosine distance)
    similarity = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
    return 1.0 - similarity 

def compute_lexical_novelty(candidate_text, reference_texts):
    """ Computes 1 - BLEU score (lower BLEU means higher novelty) """

    if not candidate_text or not reference_texts:
        return 1.0 # Max novelty if no reference

    # Tokenize texts
    candidate_tokens = candidate_text.split()
    reference_tokens = [ref.split() for ref in reference_texts if ref]

    # Max novelty if no valid references
    if not reference_tokens:
         return 1.0 

    # Compute sentence BLEU with smoothing
    sf = SmoothingFunction().method1
    try:
        bleu = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=sf)
        return 1.0 - bleu
    except ZeroDivisionError:
        return 1.0 


def compute_diversity_score(semantic_dissim, lexical_novelty):
    """ Combines semantic and lexical novelty multiplicatively """

    return semantic_dissim * lexical_novelty

def compute_macro_f1(y_true, y_pred):
    """ Computes Macro-F1 score """

    return f1_score(y_true, y_pred, average='macro', zero_division=0)

def compute_accuracy(y_true, y_pred):
    """ Computes Accuracy score """

    return accuracy_score(y_true, y_pred)

def get_class_counts(dataframe):
    """ Gets sample counts for each class """

    return Counter(dataframe['label'])

def get_class_sizes(dataframe):
    """ Gets sample counts for each class as a dictionary """

    return dataframe['label'].value_counts().to_dict()

def sample_balanced_subset(dataframe, samples_per_class=None, total_samples=None):
    """ Samples a balanced subset from a dataframe """

    if samples_per_class is None and total_samples is None:
        raise ValueError("Must specify either samples_per_class or total_samples")


    balanced_df = pd.DataFrame()
    classes = dataframe['label'].unique()
    class_counts = get_class_counts(dataframe)

    if total_samples is not None:
        samples_per_class = total_samples // len(classes)
        samples_per_class = max(1, samples_per_class)

    for cls in classes:
        class_df = dataframe[dataframe['label'] == cls]
        n_samples = min(samples_per_class, len(class_df))
        balanced_df = pd.concat([balanced_df, class_df.sample(n=n_samples, random_state=42)]) # Use fixed seed for reproducibility

    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle and reset index

def create_temporary_dataset(train_df, augmented_df):
    """ Combines original training data and augmented data """
    
    # Ensure augmented_df has 'text' and 'label' columns
    if augmented_df.empty:
        return train_df.copy()
    
    # Assuming augmented_df structure is correct (text, label)
    combined_df = pd.concat([train_df, augmented_df], ignore_index=True)
    return combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

