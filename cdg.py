import pandas as pd
import json
from tqdm import tqdm
from llm_api import call_llm
from prompts import CDG_BASIC_PROMPT, CDG_EMPHASIS_PROMPT # Choose based on need
from config import DEALTConfig

class ConditionalDiversityGenerator:
    """Generates augmented samples based on planning instructions."""

    def __init__(self, config: DEALTConfig):
        self.config = config

    def generate_sample(self, seed_sample: pd.Series, instruction: str, use_emphasis: bool = True):
        """Generates a single augmented sample for a seed and instruction."""
        
        seed_text = seed_sample[self.config.TEXT_COLUMN]
        seed_label = seed_sample[self.config.LABEL_COLUMN]

        if use_emphasis:
             prompt = CDG_EMPHASIS_PROMPT.format(
                seed_text=seed_text,
                seed_label=seed_label,
                augmentation_instruction=instruction
            )
        else:
            prompt = CDG_BASIC_PROMPT.format(
                seed_text=seed_text,
                seed_label=seed_label,
                augmentation_instruction=instruction
            )

        # Use generation temperature
        augmented_text = call_llm(
            prompt,
            model=self.config.LLM_MODEL,
            temperature=self.config.LLM_TEMP_GENERATION,
            max_tokens=self.config.LLM_MAX_TOKENS_CDG # Use CDG token limit
            )

        if augmented_text:
            return {
                self.config.TEXT_COLUMN: augmented_text.strip(),
                self.config.LABEL_COLUMN: seed_label, # Keep original label
                'seed_text': seed_text,
                'seed_label': seed_label,
                'instruction': instruction
                }
        return None

