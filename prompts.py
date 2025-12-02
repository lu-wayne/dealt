# Prompts


# LDT Prompts
LDT_IMPLICIT_THEME_PROMPT = """Instruction: Analyze the following text samples, all belonging to the class '{class_label}'.
Identify up to 3 distinct, potentially underrepresented sub-themes or specific scenarios present in these samples.
For each theme, provide a brief (1-2 sentence) description.

Class: {class_label}
Samples:
{samples_text}

Identified Sub-themes:"""


LDT_EXPLICIT_FOCUS_PROMPT = """Instruction: Given the following statistics about classes in our dataset, and knowing our goal is to augment tail classes, suggest the top {N_focus_areas} classes that most urgently need data augmentation.
Provide a brief justification for each.

Dataset Statistics:
{dataset_stats}

Current Augmentation Focus (from LDT): {explicit_tail_classes}

Suggested Top {N_focus_areas} Classes for Urgent Augmentation:""" # N_focus_areas defined in AISE


# DP Prompts
DP_BASIC_PROMPT = """Instruction: You are an expert in creative text data augmentation.
Given the original seed sentence from class '{seed_label}' and a set of diversity dimensions, generate {N_strategies} distinct and actionable augmentation instructions.
Each instruction should aim to create a new sentence that is semantically consistent with the original label but explores a different facet or variation.

Original Seed Sentence: "{seed_text}"
Label: "{seed_label}"
Diversity Dimensions to Consider: {diversity_dimensions}
Number of Strategies to Generate: {N_strategies}

Generated Augmentation Instructions:"""


DP_KNOWLEDGE_POOL_PROMPT = """Instruction: Your task is to devise {N_strategies} diverse augmentation strategies for the given seed sentence.
Leverage the provided "Knowledge Pool Concepts" to make the strategies more targeted and rich.
Ensure the generated samples would still be valid for the class '{seed_label}'.

Original Seed Sentence: "{seed_text}"
Label: "{seed_label}"
Knowledge Pool Concepts related to '{seed_label}' and '{seed_text}':
{knowledge_pool}

Diversity Dimensions to Emphasize: {diversity_dimensions_emphasize}
Number of Strategies to Generate: {N_strategies}

Generated Augmentation Instructions (incorporating knowledge pool):"""


# CDG Prompts
CDG_BASIC_PROMPT = """Instruction: Generate a new text sample based on the Original Sentence and the provided Augmentation Instruction.
The new sample MUST remain consistent with the Original Label '{seed_label}'.
Do not repeat the instruction in your output, only the generated sample.

Original Sentence: "{seed_text}"
Original Label: "{seed_label}"
Augmentation Instruction: "{augmentation_instruction}"

Generated Sample:"""

CDG_EMPHASIS_PROMPT = """Instruction: Carefully follow the Augmentation Instruction to create a new version of the Original Sentence.
Pay special attention to maintaining high plausibility and naturalness for the class '{seed_label}'.
The new sample should clearly embody the transformation described in the instruction.

Original Sentence: "{seed_text}"
Original Label: "{seed_label}"
Augmentation Instruction: "{augmentation_instruction}"
Emphasis: Ensure the generated text flows naturally and is a very believable example of '{seed_label}'.

Generated Sample:"""


# QDV Prompts
QDV_QUALITY_PROMPT = """Instruction: You are an expert annotator. Assess if the "Generated Sample" is a good and faithful augmentation of the "Original Sample" for the given "Original Label".
Consider:
1. Label Consistency: Does the Generated Sample accurately belong to the '{original_label}'?
2. Plausibility & Fluency: Is the Generated Sample grammatically correct, fluent, and sensical?

Original Sample: "{original_text}"
Original Label: "{original_label}"
Generated Sample: "{augmented_text}"

Assessment:
Label Consistent (Yes/No):
Plausible & Fluent (Yes/No):
Brief Justification (1-2 sentences if No on any, or if marginal):"""


QDV_DIVERSITY_PROMPT = """Instruction: Evaluate the "Generated Sample" in comparison to the "Original Sample".
Focus on:
1. Semantic Novelty: How different is the core meaning or scenario in the Generated Sample compared to the Original? (e.g., Minor rephrasing, Different perspective, New related scenario)
2. Label Adherence: Despite any novelty, does the Generated Sample strictly adhere to the '{original_label}'?

Original Sample: "{original_text}"
Original Label: "{original_label}"
Generated Sample: "{augmented_text}"

Evaluation:
Semantic Novelty (Describe level):
Label Adherence (Yes/No):
Overall Quality as Augmentation (Good/Acceptable/Poor):
Rationale:"""


# AISE Prompts
AISE_NEXT_FOCUS_PROMPT = """Instruction: Based on the recent proxy model performance changes and error analysis for different classes/clusters, suggest which {N_focus_areas} (classes or identified implicit clusters) should be prioritized for the next round of data augmentation.
Consider the remaining budget and aim to improve overall model robustness, especially on underperforming areas.

Performance Summary:
{performance_summary}

Error Analysis Snippets (from Logs):
{error_log_snippets}

Remaining Augmentation Budget (relative): {budget_remaining_percentage}%
Number of Focus Areas to Suggest: {N_focus_areas}

Suggested Prioritized Focus Areas for Next Round:"""



AISE_REFINE_STRATEGIES_PROMPT = """Instruction: Several augmented samples for class '{class_label}' (generated using the 'Original Augmentation Strategies') are consistently failing the QDV or leading to poor proxy model performance.
Analyze the provided examples of original texts, their failed augmentations, and the strategies used.
Suggest {N_revised_strategies} revised or new types of diversity strategies that might yield better results for this class, specifically addressing the observed failure patterns.

Class: '{class_label}'
Examples of Original Texts & Failed Augmentations:
{failed_examples}

Observed Failure Patterns (if known, else LLM infers): {summary_of_failure_patterns}
Number of Revised/New Strategies to Suggest: {N_revised_strategies}

Suggested Revised/New Diversity Strategies:"""


