import pandas as pd
from config import DEALTConfig
from proxy_model import ProxyModel # Import the proxy model class
from utils import create_temporary_dataset, compute_macro_f1, sample_balanced_subset
from llm_api import call_llm
from prompts import AISE_NEXT_FOCUS_PROMPT, AISE_REFINE_STRATEGIES_PROMPT
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdaptiveIncrementalSamplerEvaluator:
    """Manages the iterative augmentation loop, sampling, and evaluation."""

    def __init__(self, config: DEALTConfig, train_df: pd.DataFrame, val_df: pd.DataFrame, proxy_model: ProxyModel):
        self.config = config
        self.train_df = train_df
        self.val_df = val_df
        self.proxy_model = proxy_model # Initialized and potentially initially trained
        self.augmented_data_val = pd.DataFrame() # Accepted augmented samples
        self.operational_logs = [] # To record generation/validation outcomes
        self.performance_history = [] # Track performance on val_df
        self.iteration = 0
        self.initial_tail_size = 0 # To be set by DEALT based on LDT results

    def set_initial_tail_size(self, size):
        self.initial_tail_size = size
        logging.info(f"Initial target tail size for augmentation budget: {self.initial_tail_size}")

    def should_continue(self):
        """Determines if the AISE loop should continue."""

        if self.iteration >= self.config.MAX_AISE_ITERATIONS:
            logging.info(f"Stopping: Max iterations reached ({self.config.MAX_AISE_ITERATIONS})")
            return False

        current_budget_size = len(self.augmented_data_val)
        target_budget_size = self.initial_tail_size * self.config.AUGMENTATION_BUDGET_MULTIPLIER
        if current_budget_size >= target_budget_size and self.config.AUGMENTATION_BUDGET_TOTAL is None:
             logging.info(f"Stopping: Augmentation budget reached ({current_budget_size} >= {target_budget_size})")
             return False

        if self.config.AUGMENTATION_BUDGET_TOTAL is not None and current_budget_size >= self.config.AUGMENTATION_BUDGET_TOTAL:
            logging.info(f"Stopping: Total augmentation budget reached ({current_budget_size} >= {self.config.AUGMENTATION_BUDGET_TOTAL})")
            return False


        # Check for plateauing (needs at least 2 iterations)
        if self.iteration >= 1 and len(self.performance_history) >= 2:
            # Compare current performance to the previous best performance
            current_perf = self.performance_history[-1]

            # Find max performance up to the *previous* iteration
            previous_max_perf = max(self.performance_history[:-1]) if self.performance_history[:-1] else -float('inf')

            if current_perf - previous_max_perf < self.config.PLATEAU_THRESHOLD:
                 logging.info(f"Stopping: Performance plateau detected (improvement < {self.config.PLATEAU_THRESHOLD:.4f})")
                 return False


        return True

    def add_augmented_samples(self, new_samples_df: pd.DataFrame):
        """Adds newly accepted augmented samples."""

        self.augmented_data_val = pd.concat([self.augmented_data_val, new_samples_df], ignore_index=True)
        logging.info(f"Added {len(new_samples_df)} accepted samples. Total accepted: {len(self.augmented_data_val)}")

    def log_outcome(self, sample_info: dict):
        """Logs the outcome of a generation/validation attempt."""

        self.operational_logs.append(sample_info)

    def evaluate_and_refine(self, starget_df: pd.DataFrame, explicit_tails: list, implicit_tail_info: dict, diversity_dimensions: list):
        """Evaluates current performance and refines strategy."""
        
        self.iteration += 1
        logging.info(f"--- Starting AISE Iteration {self.iteration} ---")

        # 1. Lightweight Proxy Performance Check
        combined_train_df = create_temporary_dataset(self.train_df, self.augmented_data_val)

        logging.info(f"Training proxy model on combined dataset ({len(combined_train_df)} samples)...")
        # Use fewer epochs for quick check in AISE iterations
        self.proxy_model.train(combined_train_df, self.val_df, epochs=2) # Reduced epochs

        # Evaluate on validation set
        logging.info("Evaluating proxy model on validation set...")
        val_results = self.proxy_model.evaluate(
             self.proxy_model.tokenizer(self.val_df[self.config.TEXT_COLUMN].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=self.config.BERT_MAX_LEN).to(self.proxy_model.device),
             self.val_df[self.config.LABEL_COLUMN].tolist() # Need original labels here
             ) # Pass DataLoader or encoded tensors + labels
        # Correct way to pass data to evaluate: need a DataLoader
        val_texts, val_labels_encoded = self.proxy_model.prepare_data(self.val_df, fit_encoder=False)
        val_dataset = TextDataset(val_texts, val_labels_encoded, self.proxy_model.tokenizer, self.config.BERT_MAX_LEN)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BERT_BATCH_SIZE)
        val_results = self.proxy_model.evaluate(val_loader)


        current_macro_f1 = val_results['macro_f1']
        self.performance_history.append(current_macro_f1)
        logging.info(f"AISE Iteration {self.iteration} Val Macro-F1: {current_macro_f1:.4f}")

        # 2. Diagnostic Analysis (Error Analysis & Performance Summary)
        performance_summary = self._analyze_performance(val_results, explicit_tails, implicit_tail_info)
        error_log_snippets = self._generate_error_snippets() # Snippets from logs

        # 3. Next-Iteration Strategy Refinement
        next_starget_df = self._select_next_target_samples(starget_df, performance_summary, self.operational_logs)

        # Parameter Update (Strategy Refinement)
        revised_diversity_dimensions = self._refine_diversity_strategies(performance_summary, error_log_snippets, diversity_dimensions)

        logging.info(f"--- Finished AISE Iteration {self.iteration} ---")

        return next_starget_df, revised_diversity_dimensions

    def _analyze_performance(self, val_results, explicit_tails, implicit_tail_info):
        """Analyzes performance by class/cluster on the validation set."""

        performance_summary = {}
        true_labels = val_results['true_labels']
        predictions = val_results['predictions']

        # Analyze performance per class
        class_performance = {}
        unique_classes = np.unique(true_labels)
        for cls in unique_classes:
            cls_indices = [i for i, label in enumerate(true_labels) if label == cls]
            if not cls_indices:
                continue
            cls_true = [true_labels[i] for i in cls_indices]
            cls_pred = [predictions[i] for i in cls_indices]
            macro_f1 = compute_macro_f1(cls_true, cls_pred)
            accuracy = compute_accuracy(cls_true, cls_pred)
            error_rate = 1.0 - accuracy
            class_performance[cls] = {'macro_f1': macro_f1, 'error_rate': error_rate, 'count': len(cls_indices)}

        # Determine performance delta if history exists
        prev_val_results = None

        # Placeholder for detailed class/cluster summary for LLM prompt
        llm_summary_str = ""
        for cls, stats in class_performance.items():
             llm_summary_str += f"- Class {cls}: F1: {stats['macro_f1']:.4f}, Error Rate: {stats['error_rate']:.4f}, Count: {stats['count']}\n"


        return {'class_performance': class_performance, 'llm_summary_str': llm_summary_str}


    def _generate_error_snippets(self):
        """Generates snippets from operational logs highlighting failures."""

        failed_logs = [log for log in self.operational_logs if not log.get('accepted')]

        # Select a few examples of failed generation/validation
        snippets = []
        for log in failed_logs[-5:]: # Take last 5 failed logs as snippets
             snippets.append(f"Seed: '{log.get('seed_text')}'\nInstruction: '{log.get('instruction')}'\nGenerated: '{log.get('augmented_text')}'\nReason: {log.get('reason')}\n")
        return "\n---\n".join(snippets)

    def _select_next_target_samples(self, initial_starget_df: pd.DataFrame, performance_summary: dict, operational_logs: list):
        """Selects samples for the next round of augmentation."""

        underperforming_classes = sorted(
            performance_summary['class_performance'].items(),
            key=lambda item: item[1]['macro_f1'] # Sort by Macro-F1
        )
        # Select the top N_FOCUS_AREAS worst performing classes (or fewer if less exist)
        focus_classes = [cls for cls, stats in underperforming_classes][:self.config.N_FOCUS_AREAS]
        logging.info(f"Focus classes for next iteration: {focus_classes}")


        # Filter the initial Starget_df to include samples from focus classes
        next_starget_df = initial_starget_df[initial_starget_df[self.config.LABEL_COLUMN].isin(focus_classes)].copy()

        failed_seeds_info = [(log['seed_text'], log['seed_label']) for log in operational_logs if not log['accepted']]
        failed_seeds_df = pd.DataFrame(failed_seeds_info, columns=[self.config.TEXT_COLUMN, self.config.LABEL_COLUMN])

        # Merge to find which samples in next_starget_df were 'failed seeds'
        next_starget_df['was_failed_seed'] = next_starget_df.apply(
            lambda row: (row[self.config.TEXT_COLUMN], row[self.config.LABEL_COLUMN]) in failed_seeds_info, axis=1
        )

        # Prioritize failed seeds first, then other samples in focus classes
        next_starget_df = next_starget_df.sort_values(by='was_failed_seed', ascending=False).drop(columns='was_failed_seed')

        logging.info(f"Selected {len(next_starget_df)} target samples for next iteration.")

        return next_starget_df

    def _refine_diversity_strategies(self, performance_summary: dict, error_log_snippets: str, current_diversity_dimensions: list):
        """Refines diversity strategies using LLM based on performance analysis and failures."""

        failed_examples_for_prompt = []
        # Select a few examples from logs that were NOT accepted
        failed_logs = [log for log in self.operational_logs if not log.get('accepted')]
        for log in failed_logs[-3:]: # Take last 3 failed logs for the prompt
             failed_examples_for_prompt.append(f"Original: '{log.get('seed_text')}'\nFailed Aug: '{log.get('augmented_text')}'\nStrategy Used: '{log.get('instruction')}'\nQDV Feedback/Error: {log.get('reason')}\n")

        if not failed_examples_for_prompt:
             logging.info("No failed examples to refine strategies.")
             return current_diversity_dimensions # No refinement needed if no failures

        failed_examples_str = "\n---\n".join(failed_examples_for_prompt)


        target_class = failed_logs[-1].get('seed_label', 'N/A') if failed_logs else 'N/A'

        prompt = AISE_REFINE_STRATEGIES_PROMPT.format(
            class_label=target_class,
            failed_examples=failed_examples_str,
            summary_of_failure_patterns="Analysis based on provided examples.", # LLM will infer patterns
            N_revised_strategies=self.config.N_REVISED_STRATEGIES
        )

        logging.info("Refining diversity strategies using LLM...")
        response = call_llm(
            prompt,
            model=self.config.LLM_MODEL,
            temperature=self.config.LLM_TEMP_PLANNING, # Use planning temp
            max_tokens=self.config.LLM_MAX_TOKENS_DP # Use DP token limit for strategies
            )

        if response:
            revised_strategies = [line.strip() for line in response.strip().split('\n') if line.strip() and line[0].isdigit()]
            logging.info(f"LLM suggested revised strategies: {revised_strategies}")

            if revised_strategies:
                return revised_strategies 

        logging.warning("LLM strategy refinement failed or returned no suggestions.")
        return current_diversity_dimensions

