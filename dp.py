import pandas as pd
import json
from tqdm import tqdm
from llm_api import call_llm
from prompts import DP_KNOWLEDGE_POOL_PROMPT, DP_BASIC_PROMPT
from config import DEALTConfig

class DiversityPlanner:
    """ Plans diverse augmentation strategies using LLM """

    def __init__(self, config: DEALTConfig):
        self.config = config

    def generate_knowledge_pool(self, seed_sample: pd.Series):
        """ Generates knowledge pool concepts using LLM """

        prompt = f"""List concepts, themes, or keywords related to the following text and its label.
Text: "{seed_sample[self.config.TEXT_COLUMN]}"
Label: "{seed_sample[self.config.LABEL_COLUMN]}"
List of Concepts:"""

        # Use slightly lower temperature for factual concepts
        response = call_llm(
            prompt,
            model=self.config.LLM_MODEL,
            temperature=0.5,
            max_tokens=self.config.LLM_MAX_TOKENS_DP # Use DP token limit
            )
        if response:
            return response.strip()
        return ""

    def generate_plan(self, seed_sample: pd.Series, diversity_dimensions: list = None, use_knowledge_pool: bool = True, N_strategies: int = None):
        """ Generates N diverse augmentation strategies for a seed sample """

        seed_text = seed_sample[self.config.TEXT_COLUMN]
        seed_label = seed_sample[self.config.LABEL_COLUMN]
        dimensions = diversity_dimensions if diversity_dimensions is not None else self.config.DIVERSITY_DIMENSIONS
        n_strategies = N_strategies if N_strategies is not None else self.config.N_STRATEGIES

        knowledge_pool = ""
        if use_knowledge_pool and self.config.USE_KNOWLEDGE_POOL:
            knowledge_pool = self.generate_knowledge_pool(seed_sample)
            # Format for prompt
            knowledge_pool_formatted = "\n".join([f"- {line.strip()}" for line in knowledge_pool.split('\n') if line.strip()])
            prompt = DP_KNOWLEDGE_POOL_PROMPT.format(
                N_strategies=n_strategies,
                seed_text=seed_text,
                seed_label=seed_label,
                knowledge_pool=knowledge_pool_formatted,
                diversity_dimensions_emphasize=",".join(dimensions)
            )
        else:
             prompt = DP_BASIC_PROMPT.format(
                N_strategies=n_strategies,
                seed_text=seed_text,
                seed_label=seed_label,
                diversity_dimensions=",".join(dimensions)
            )

        # Use CoT implicitily by prompt structure and LLM capabilities
        response = call_llm(
            prompt,
            model=self.config.LLM_MODEL,
            temperature=self.config.LLM_TEMP_PLANNING,
            max_tokens=self.config.LLM_MAX_TOKENS_DP # Use DP token limit
        )

        if response:
            instructions = [line.strip() for line in response.strip().split('\n') if line.strip() and line[0].isdigit()]
            return instructions[:n_strategies] # Return up to N strategies
        return []

