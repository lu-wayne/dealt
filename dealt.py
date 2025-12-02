import pandas as pd
from config import DEALTConfig
from ldt import LongTailDistributionDetector
from dp import DiversityPlanner
from cdg import ConditionalDiversityGenerator
from qdv import QualityDiversityValidator
from aise import AdaptiveIncrementalSamplerEvaluator
from proxy_model import ProxyModel, TextDataset # Import ProxyModel and TextDataset for final training
from utils import load_data, sample_balanced_subset, compute_macro_f1, compute_accuracy, create_temporary_dataset
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import logging
import os
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DEALT:
    """ The main DEALT framework orchestrator """

    def __init__(self, config: DEALTConfig, dataset_name: str):
        self.config = config
        self.dataset_name = dataset_name
        self.train_df = load_data(os.path.join(config.DATA_DIR, dataset_name, config.TRAIN_FILE))
        self.val_df = load_data(os.path.join(config.DATA_DIR, dataset_name, config.VAL_FILE))
        self.test_df = load_data(os.path.join(config.DATA_DIR, dataset_name, config.TEST_FILE))

        # Initialize modules
        self.ldt = LongTailDistributionDetector(config, self.train_df)
        self.ldt.set_dataset_name(dataset_name) # Set dataset name for LDT config

        # Proxy model initial training (on a balanced subset)
        logging.info("Training initial proxy model...")
        proxy_train_subset = sample_balanced_subset(self.train_df, total_samples=int(len(self.train_df) * config.PROXY_BALANCED_SUBSET_PERCENT / 100))
        self.proxy_model = ProxyModel(num_labels=len(self.train_df[config.LABEL_COLUMN].unique()), model_path=config.PROXY_MODEL_PATH)

        # Fit encoder on the full training data labels to ensure all possible labels are included
        self.proxy_model.prepare_data(self.train_df, fit_encoder=True)
        self.proxy_model.train(proxy_train_subset, self.val_df, epochs=3) # Use a few epochs for initial training

        self.dp = DiversityPlanner(config)
        self.cdg = ConditionalDiversityGenerator(config)
        self.qdv = QualityDiversityValidator(config, self.proxy_model) # Pass the trained proxy model
        self.aise = AdaptiveIncrementalSamplerEvaluator(config, self.train_df, self.val_df, self.proxy_model) # Pass the trained proxy model

        self.augmented_data_val = pd.DataFrame() # Accumulate accepted samples

    def run(self):
        """ Runs the DEALT data augmentation process """

        logging.info("--- Starting DEALT Augmentation Process ---")

        # 1. Long-tail Distribution Detection
        logging.info("Identifying target samples (explicit and implicit tails)...")
        starget_df_initial, explicit_tails, implicit_tail_info = self.ldt.identify_target_samples()
        logging.info(f"Identified {len(starget_df_initial)} target samples for augmentation.")
        self.aise.set_initial_tail_size(len(starget_df_initial)) 
        current_starget_df = starget_df_initial.copy()
        current_diversity_dimensions = self.config.DIVERSITY_DIMENSIONS # Initial dimensions

        while self.aise.should_continue():
            logging.info(f"AISE Iteration {self.aise.iteration + 1}: Generating augmented data...")
            newly_generated_samples = []
            iteration_logs = [] # Logs for this iteration

            if current_starget_df.empty:
                 logging.warning("Current target samples is empty. Stopping AISE loop.")
                 break

            for _, seed_sample in tqdm(current_starget_df.iterrows(), total=len(current_starget_df), desc=f"AISE Iter {self.aise.iteration + 1} Generating"):
                seed_text = seed_sample[self.config.TEXT_COLUMN]
                seed_label = seed_sample[self.config.LABEL_COLUMN]

                # 2. Diversity Planning
                plan_instructions = self.dp.generate_plan(seed_sample, diversity_dimensions=current_diversity_dimensions, use_knowledge_pool=self.config.USE_KNOWLEDGE_POOL, N_strategies=self.config.N_STRATEGIES)
                if not plan_instructions:
                     logging.warning(f"No plan generated for seed: '{seed_text}'. Skipping.")
                     continue

                # 3. Conditional Diversity Generation
                for instruction in plan_instructions:
                     # Use emphasis 
                     raw_sample = self.cdg.generate_sample(seed_sample, instruction, use_emphasis=True)

                     if raw_sample:
                         # 4. Quality and Diversity Validation
                         accepted, reason = self.qdv.validate(raw_sample, seed_sample, self.augmented_data_val)
                         raw_sample['accepted'] = accepted
                         raw_sample['reason'] = reason
                         iteration_logs.append(raw_sample) # Log outcome
                         if accepted:
                             newly_generated_samples.append(raw_sample)
                             logging.debug(f"Accepted: {raw_sample['text']} (Seed: {seed_text})")
                         else:
                             logging.debug(f"Rejected: {raw_sample.get('text')} (Seed: {seed_text}, Reason: {reason})")
                     else:
                         raw_sample = {'seed_text': seed_text, 'seed_label': seed_label, 'instruction': instruction, 'augmented_text': None, 'accepted': False, 'reason': 'LLM generation failed'}
                         iteration_logs.append(raw_sample)
                         logging.warning(f"LLM generation failed for seed: '{seed_text}' with instruction: '{instruction}'.")


            newly_generated_df = pd.DataFrame(newly_generated_samples)
            self.aise.add_augmented_samples(newly_generated_df) # Add to cumulative accepted data
            self.aise.operational_logs.extend(iteration_logs) # Add logs for this iteration

            # 5. Adaptive Incremental Sampler and Evaluator
            if self.aise.should_continue(): # Re-check condition before evaluating
                 current_starget_df, current_diversity_dimensions = self.aise.evaluate_and_refine(starget_df_initial, explicit_tails, implicit_tail_info, current_diversity_dimensions)
            else:
                 logging.info("AISE loop stopping after generation due to budget or max iterations.")


        logging.info("--- DEALT Augmentation Process Finished ---")
        logging.info(f"Total augmented samples generated and accepted: {len(self.augmented_data_val)}")

        # Final combined training data
        final_train_df = create_temporary_dataset(self.train_df, self.augmented_data_val)

        return final_train_df

    def train_and_evaluate_final_model(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Trains the final BERT classifier and evaluates it."""
        logging.info("--- Starting Final BERT Model Training ---")

        # Prepare data for BERT
        label_encoder = LabelEncoder()
        train_labels_encoded = label_encoder.fit_transform(train_df[self.config.LABEL_COLUMN].tolist())
        test_labels_encoded = label_encoder.transform(test_df[self.config.LABEL_COLUMN].tolist())
        num_labels = len(label_encoder.classes_)

        # Split combined train_df into training and validation for BERT fine-tuning
        combined_train_texts = train_df[self.config.TEXT_COLUMN].tolist()
        combined_train_labels = train_df[self.config.LABEL_COLUMN].tolist()
        combined_train_labels_encoded = label_encoder.transform(combined_train_labels)


        train_texts, bert_val_texts, train_labels, bert_val_labels = train_test_split(
            combined_train_texts, combined_train_labels_encoded,
            test_size=self.config.BERT_VALIDATION_PERCENT / 100,
            random_state=self.config.RANDOM_SEED,
            stratify=combined_train_labels_encoded # Stratify split to maintain label distribution
        )

        # Create datasets
        tokenizer = BertTokenizer.from_pretrained(self.config.BERT_MODEL_PATH)
        train_dataset = TextDataset(train_texts, label_encoder.inverse_transform(train_labels), tokenizer, self.config.BERT_MAX_LEN)
        bert_val_dataset = TextDataset(bert_val_texts, label_encoder.inverse_transform(bert_val_labels), tokenizer, self.config.BERT_MAX_LEN)
        test_dataset = TextDataset(test_df[self.config.TEXT_COLUMN].tolist(), test_df[self.config.LABEL_COLUMN].tolist(), tokenizer, self.config.BERT_MAX_LEN) # Use original labels

        # Load BERT model
        model = BertForSequenceClassification.from_pretrained(self.config.BERT_MODEL_PATH, num_labels=num_labels)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=f'./results/{self.dataset_name}_dealt_finetuning',
            num_train_epochs=self.config.BERT_EPOCHS,
            per_device_train_batch_size=self.config.BERT_BATCH_SIZE,
            per_device_eval_batch_size=self.config.BERT_BATCH_SIZE,
            learning_rate=self.config.BERT_LEARNING_RATE,
            warmup_ratio=0.1, # 10% warmup 
            weight_decay=0.01, 
            evaluation_strategy="epoch", 
            save_strategy="epoch", 
            load_best_model_at_end=True, 
            metric_for_best_model="macro_f1", 
            greater_is_better=True,
            logging_dir='./logs',
            logging_steps=100,
            report_to="none", 
            seed=self.config.RANDOM_SEED
        )

        # Metrics for Trainer
        def compute_metrics(p):
            preds = p.predictions.argmax(-1)
            labels = p.label_ids
            macro_f1 = compute_macro_f1(labels, preds)
            accuracy = compute_accuracy(labels, preds)
            return {"accuracy": accuracy, "macro_f1": macro_f1}

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=bert_val_dataset, # early stopping
            compute_metrics=compute_metrics,
        )

        # Train the model
        trainer.train()

        logging.info("--- Final BERT Model Training Finished ---")

        # Evaluate on test set
        logging.info("--- Evaluating Final Model on Test Set ---")
        test_results = trainer.evaluate(test_dataset)
        logging.info(f"Test Results: {test_results}")

        return test_results

