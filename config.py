import os
import pandas as pd
import json
from tqdm import tqdm

class DEALTConfig:
    """Central configuration for DEALT framework """

    # Data Paths
    DATA_DIR = './data'
    TRAIN_FILE = 'train.csv'
    VAL_FILE = 'val.csv'
    TEST_FILE = 'test.csv'
    TEXT_COLUMN = 'text'
    LABEL_COLUMN = 'label'


    # LLM Configuration 
    LLM_MODEL = "gpt-4o-mini" # Base model for all LLM calls
    LLM_TEMP_PLANNING = 0.7   # Temperature for Diversity Planning
    LLM_TEMP_GENERATION = 0.7 # Temperature for Conditional Diversity Generation
    LLM_TEMP_VALIDATION = 0.1 # Lower temp for validation to be more deterministic
    LLM_MAX_TOKENS_DP = 256   # Max tokens for DP output
    LLM_MAX_TOKENS_CDG = 512  # Max tokens for CDG output
    LLM_MAX_TOKENS_QDV = 128  # Max tokens for QDV judgment


    # LDT (Long-tail Distribution Detector) Configuration
    K_EXPLICIT_THRESHOLD_PERCENTILE = 25 # %ile of class sizes
    K_EXPLICIT_THRESHOLD_MIN_COUNT = 10 # Minimum count for explicit tail
    K_IMPLICIT_THRESHOLD_COUNT = 5 # Minimum count for implicit tail cluster size 
    K_IMPLICIT_THRESHOLD_PERCENTAGE_AVG = 10 # %ile of average class size for implicit tail 
    DBSCAN_MIN_SAMPLES = 3 # DBSCAN min_samples
    DBSCAN_EPS = {'default': 0.4} # DBSCAN eps, can be tuned per dataset. Add dataset names as keys if specific values needed.
    K_REP = 1 # Number of representative samples per implicit cluster
    EMBEDDING_MODEL_PATH = 'all-MiniLM-L6-v2' # Sentence-BERT model


    # DP (Diversity Planning) Configuration
    N_STRATEGIES = 3 # Number of diversity strategies per seed
    DIVERSITY_DIMENSIONS = [ 
        'Lexical variation',
        'Syntactic restructuring',
        'Semantic scenario change/expansion',
        'Sentiment/tone modulation (where applicable)'
    ]
    USE_KNOWLEDGE_POOL = True # Use knowledge pool in DP 

    # CDG (Conditional Diversity Generation) Configuration
    SAMPLES_PER_INSTRUCTION = 1 # Number of augmented samples per instruction


    # QDV (Quality and Diversity Validator) Configuration
    PROXY_MODEL_PATH = './pretrain_models/distilbert-base-cased' # Path for proxy model
    PROXY_BALANCED_SUBSET_PERCENT = 20 # % of original train data for proxy initial training
    T_CONF = 0.7 # Proxy confidence threshold
    T_DIV = 0.3 # Composite diversity threshold


    # AISE (Adaptive Incremental Sampler and Evaluator) Configuration
    AUGMENTATION_BUDGET_MULTIPLIER = 2 # Target size = multiplier * initial_tail_size
    # Or a fixed total number: AUGMENTATION_BUDGET_TOTAL = 2000 # Example fixed budget
    MAX_AISE_ITERATIONS = 3 # Max iterative rounds
    PLATEAU_THRESHOLD = 0.005 # Macro-F1 improvement threshold for plateauing
    N_FOCUS_AREAS = 3 # Number of focus areas to suggest in AISE 
    N_REVISED_STRATEGIES = 3 # Number of revised strategies to suggest in AISE


    # Downstream Classifier Configuration (BERT-base-cased)
    BERT_MODEL_PATH = './pretrain_models/bert-base-cased' # Path for BERT model
    BERT_LEARNING_RATE = 2e-5 
    BERT_BATCH_SIZE = 16
    BERT_EPOCHS = 5 
    BERT_MAX_LEN = 256 # Max sequence length 
    BERT_VALIDATION_PERCENT = 10 # % of train data for validation split during BERT fine-tuning


    # Other
    RANDOM_SEED = 42 
    LOGGING_LEVEL = 'INFO' 

    def get_dataset_config(self, dataset_name):
        """Get dataset-specific configurations if any."""
        
        config = {
            'k_explicit': None,
            'k_implicit': None,
            'dbscan_eps': self.DBSCAN_EPS.get(dataset_name, self.DBSCAN_EPS['default']),
            'proxy_balanced_subset_size': None, 
        }

        return config

