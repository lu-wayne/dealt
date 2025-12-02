import pandas as pd
from llm_api import call_llm
from prompts import QDV_QUALITY_PROMPT, QDV_DIVERSITY_PROMPT
from config import DEALTConfig
from utils import compute_embeddings, compute_diversity_score, compute_semantic_dissimilarity, compute_lexical_novelty
import numpy as np
import torch

class QualityDiversityValidator:
    """ Validates quality and diversity of generated samples """

    def __init__(self, config: DEALTConfig, proxy_model):
        self.config = config
        self.proxy_model = proxy_model
        self.embedding_model = compute_embeddings

    def validate(self, raw_sample: dict, seed_sample: pd.Series, accepted_augmentations_df: pd.DataFrame):
        """ Validates a raw augmented sample hierarchically """

        aug_text = raw_sample.get(self.config.TEXT_COLUMN)
        aug_label = raw_sample.get(self.config.LABEL_COLUMN)
        seed_text = seed_sample[self.config.TEXT_COLUMN]
        seed_label = seed_sample[self.config.LABEL_COLUMN]

        if not aug_text or not aug_label:
            return False, "Empty sample"

        # 1. Rapid Quality Screening (Proxy Model)
        # Use predict to get label and confidence
        predicted_labels, confidences = self.proxy_model.predict([aug_text])
        proxy_label = predicted_labels[0]
        proxy_confidence = confidences[0]

        # Check label consistency
        label_consistent = (proxy_label == aug_label)

        # Embed the new sample and seed
        aug_embedding = self.embedding_model(aug_text)
        seed_embedding = self.embedding_model(seed_text)

        # Need embeddings of existing accepted samples for diversity check
        existing_aug_texts = accepted_augmentations_df[self.config.TEXT_COLUMN].tolist()

        # Include the seed text as a reference for diversity
        reference_texts = [seed_text] + existing_aug_texts

        # 2. Diversity Assessment
        semantic_dissim_seed = compute_semantic_dissimilarity(aug_embedding, seed_embedding)

        # Calculate average semantic dissimilarity to existing accepted samples
        semantic_dissim_existing = 0.0
        if not accepted_augmentations_df.empty:
            existing_aug_embeddings = self.embedding_model(existing_aug_texts)
            if existing_aug_embeddings.shape[0] > 0 and aug_embedding.shape[0] > 0:
                 # Compute pairwise distances between the new sample and all existing ones
                 pairwise_dissim = 1 - cosine_similarity(aug_embedding, existing_aug_embeddings)[0]
                 semantic_dissim_existing = np.mean(pairwise_dissim)

        lexical_novelty = compute_lexical_novelty(aug_text, reference_texts) # Use seed + existing as references

        composite_diversity_score = compute_diversity_score(semantic_dissim_seed, lexical_novelty)

        # Check if below threshold for triggering LLM
        trigger_llm = False
        reasons = []
        if proxy_confidence < self.config.T_CONF:
            trigger_llm = True
            reasons.append(f"Low proxy confidence ({proxy_confidence:.2f} < {self.config.T_CONF})")

        # Re-evaluating diversity check for triggering LLM based
        min_semantic_dissim_to_refs = semantic_dissim_seed
        if not accepted_augmentations_df.empty and aug_embedding.shape[0] > 0:
             existing_aug_embeddings = self.embedding_model(existing_aug_texts)
             if existing_aug_embeddings.shape[0] > 0:
                 pairwise_dissim_to_existing = 1 - cosine_similarity(aug_embedding, existing_aug_embeddings)[0]
                 min_semantic_dissim_to_refs = min(min_semantic_dissim_to_refs, np.min(pairwise_dissim_to_existing))

        if min_semantic_dissim_to_refs < self.config.T_DIV: # Using semantic dissim to refs for T_DIV
             trigger_llm = True
             reasons.append(f"Low semantic dissimilarity to references ({min_semantic_dissim_to_refs:.2f} < {self.config.T_DIV})")


        # 3. LLM-Powered Adjudication (if triggered)
        if trigger_llm:
            # Quality Check Prompt
            quality_prompt = QDV_QUALITY_PROMPT.format(
                original_text=seed_text,
                original_label=seed_label,
                augmented_text=aug_text
            )
            quality_response = call_llm(
                quality_prompt,
                model=self.config.LLM_MODEL,
                temperature=self.config.LLM_TEMP_VALIDATION, # Lower temp for validation
                max_tokens=self.config.LLM_MAX_TOKENS_QDV # Use QDV token limit
                )
            if quality_response and "Label Consistent (Yes/No): Yes" in quality_response and "Plausible & Fluent (Yes/No): Yes" in quality_response:
                 return True, "Accepted by LLM adjudication"
            else:
                 return False, f"Rejected by LLM adjudication ({quality_response})"


        if label_consistent and proxy_confidence >= self.config.T_CONF and composite_diversity_score >= self.config.T_DIV:
             return True, "Accepted by proxy and diversity metrics"
        else:
            reject_reason = ""
            if not label_consistent: reject_reason += "Proxy label inconsistent. "
            if proxy_confidence < self.config.T_CONF: reject_reason += f"Low proxy confidence ({proxy_confidence:.2f} < {self.config.T_CONF}). "
            if composite_diversity_score < self.config.T_DIV: reject_reason += f"Low composite diversity ({composite_diversity_score:.2f} < {self.config.T_DIV}). "
            return False, f"Rejected by proxy or metrics ({reject_reason.strip()})"
