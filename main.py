import argparse
from config import DEALTConfig
from dealt import DEALT
import os
import pandas as pd
import json
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Run DEALT for long-tail text classification.")
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset (should correspond to a directory in ./data)")


    args = parser.parse_args()

    config = DEALTConfig()

    # Ensure directories exist
    os.makedirs(os.path.join(config.DATA_DIR, args.dataset), exist_ok=True)
    os.makedirs('./pretrain_models/bert-base-cased', exist_ok=True) 
    os.makedirs('./pretrain_models/distilbert-base-cased', exist_ok=True) 
    os.makedirs(f'./results/{args.dataset}_dealt_finetuning', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)

    # Initialize and run DEALT
    dealt_framework = DEALT(config, args.dataset)

    # Run the augmentation process
    final_train_df = dealt_framework.run()

    # Train and evaluate the final downstream model
    dealt_framework.train_and_evaluate_final_model(final_train_df, dealt_framework.test_df)

if __name__ == "__main__":
    main()
