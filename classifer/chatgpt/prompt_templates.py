ZERO_SHOT_PROMPT_TEMPLATE = """
You are tasked with classifying a cancer patient into treatments.
You will have the following information available:
1. The patient's cancer type.
2. A pair of the patient's actionable genes.
3. The list of possible treatments, delimited with square brackets. Not all treatments are correct.

Perform the following tasks:
1. Assign this patient to at least 1 and up to {max_treatments} treatments based on probabilities.
2. Provide your response in a list format. Do no provide any additional information except for the treatments.

List of treatments: {treatments}

A patient has {cancer_type} with the following actionable genes: {genes}.
What treatments are available for this patient?
"""

FEW_SHOT_PROMPT_TEMPLATE = """
You are tasked with classifying a cancer patient into treatments.
You will have the following information available:
1. The patient's cancer type.
2. A pair of the patient's actionable genes.
3. The list of possible treatments, delimited with square brackets. Not all treatments are correct.
4. A few prior examples of classifications.

Perform the following tasks:
1. Assign this patient to at least 1 and up to {max_treatments} treatments based on probabilities.
2. Provide your response in a list format. Do no provide any additional information except for the treatments.

List of treatments: {treatments}

{examples}

A patient has {cancer_type} with the following actionable genes: {genes}.
What treatments are available for this patient?
"""

ZERO_SHOT_COT_PROMPT_TEMPLATE = """
You are tasked with classifying a cancer patient into treatments.
You will have the following information available:
1. The patient's cancer type.
2. A pair of the patient's actionable genes.
3. The list of possible treatments, delimited with square brackets. Not all treatments are correct.

Perform the following tasks:
1. Assign this patient to at least 1 and up to {max_treatments} treatments based on probabilities.
2. Think step by step about which treatment is best and reason about your decision.
3. Provide the treatments in a list format after your reasoning steps.

List of treatments: {treatments}

A patient has {cancer_type} with the following actionable genes: {genes}.
What treatments are available for this patient?
"""

FEW_SHOT_COT_PROMPT_TEMPLATE = """
You are tasked with classifying a cancer patient into treatments.
You will have the following information available:
1. The patient's cancer type.
2. A pair of the patient's actionable genes.
3. The list of possible treatments, delimited with square brackets. Not all treatments are correct.
4. A few prior examples of classifications.

Perform the following tasks:
1. Assign this patient to at least 1 and up to {max_treatments} treatments based on probabilities.
2. Think step by step about which treatment is best and reason about your decision.
3. Provide the treatments in a list format after your reasoning steps.

List of treatments: {treatments}

{examples}

A patient has {cancer_type} with the following actionable genes: {genes}.
What treatments are available for this patient?
"""

ZERO_SHOT_WITH_SOURCES_PROMPT_TEMPLATE = """
You are tasked with classifying a cancer patient into treatments.
You will have the following information available:
1. The patient's cancer type.
2. A pair of the patient's actionable genes.
3. The list of pairs of possible treatments and abstracts outlining the source of the treatment. Each treatment and source pair is delimited by triple quotes. Not all treatments are correct.

Perform the following tasks:
1. Assign this patient to at least 1 and up to {max_treatments} treatments based on probabilities.
2. Provide your response in a list format. Do no provide any additional information except for the treatments.

List of treatments and sources: {treatments_and_sources}

A patient has {cancer_type} with the following actionable genes: {genes}.
What treatments are available for this patient?
"""

FEW_SHOT_WITH_SOURCES_PROMPT_TEMPLATE = """
You are tasked with classifying a cancer patient into treatments.
You will have the following information available:
1. The patient's cancer type.
2. A pair of the patient's actionable genes.
3. The list of pairs of possible treatments and abstracts outlining the source of the treatment. Each treatment and source pair is delimited by triple quotes. Not all treatments are correct.
4. A few prior examples of classifications.

Perform the following tasks:
1. Assign this patient to at least 1 and up to {max_treatments} treatments based on probabilities.
2. Provide your response in a list format. Do no provide any additional information except for the treatments.

List of treatments and sources: {treatments_and_sources}

{examples}

A patient has {cancer_type} with the following actionable genes: {genes}.
What treatments are available for this patient?
"""

ZERO_SHOT_COT_WITH_SOURCES_PROMPT_TEMPLATE = """
You are tasked with classifying a cancer patient into treatments.
You will have the following information available:
1. The patient's cancer type.
2. A pair of the patient's actionable genes.
3. The list of pairs of possible treatments and abstracts outlining the source of the treatment. Each treatment and source pair is delimited by triple quotes. Not all treatments are correct.

Perform the following tasks:
1. Assign this patient to at least 1 and up to {max_treatments} treatments based on probabilities.
2. Think step by step about which treatment is best and reason about your decision.
3. Provide the treatments in a list format after your reasoning steps.

List of treatments and sources: {treatments_and_sources}

A patient has {cancer_type} with the following actionable genes: {genes}.
What treatments are available for this patient?
"""

FEW_SHOT_COT_WITH_SOURCES_PROMPT_TEMPLATE = """
You are tasked with classifying a cancer patient into treatments.
You will have the following information available:
1. The patient's cancer type.
2. A pair of the patient's actionable genes.
3. The list of pairs of possible treatments and abstracts outlining the source of the treatment. Each treatment and source pair is delimited by triple quotes. Not all treatments are correct.
4. A few prior examples of classifications.

Perform the following tasks:
1. Assign this patient to at least 1 and up to {max_treatments} treatments based on probabilities.
2. Think step by step about which treatment is best and reason about your decision.
3. Provide the treatments in a list format after your reasoning steps.

List of treatments and sources: {treatments_and_sources}

{examples}

A patient has {cancer_type} with the following actionable genes: {genes}.
What treatments are available for this patient?
"""

ZERO_SHOT_TREATMENT_ONLY_PROMPT_TEMPLATE = """
You are tasked with determining the treatment type for a journal abstract.
You will have the following information available:
1. The journal abstract.
2. The list of possible treatments, delimited with square brackets. Only one treatment is correct.

Perform the following tasks:
1. Classify the journal abstract into a treatment.
2. Provide the treatment as a response. Do no provide any additional information except for the treatment.

List of treatments: {treatments}

What treatment is this journal abstract about: {source}?
"""

FEW_SHOT_TREATMENT_ONLY_PROMPT_TEMPLATE = """
You are tasked with determining the treatment type for a journal abstract.
You will have the following information available:
1. The journal abstract.
2. The list of possible treatments, delimited with square brackets. Only one treatment is correct.
4. A few prior examples of classifications.

Perform the following tasks:
1. Classify the journal abstract into a treatment.
2. Provide the treatment as a response. Do no provide any additional information except for the treatment.

List of treatments: {treatments}

{examples}

What treatment is this journal abstract about: {source}?
"""

ZERO_SHOT_TREATMENT_ONLY_COT_PROMPT_TEMPLATE = """
You are tasked with determining the treatment type for a journal abstract.
You will have the following information available:
1. The journal abstract.
2. The list of possible treatments, delimited with square brackets. Only one treatment is correct.

Perform the following tasks:
1. Classify the journal abstract into a treatment.
2. Think step by step about which treatment the abstract refers to and reason about your decision.
3. Provide the treatment as a response after your reasoning steps.

List of treatments: {treatments}

What treatment is this journal abstract about: {source}?
"""

FEW_SHOT_TREATMENT_ONLY_COT_PROMPT_TEMPLATE = """
You are tasked with determining the treatment type for a journal abstract.
You will have the following information available:
1. The journal abstract.
2. The list of possible treatments, delimited with square brackets. Only one treatment is correct.
4. A few prior examples of classifications.

Perform the following tasks:
1. Classify the journal abstract into a treatment.
2. Think step by step about which treatment the abstract refers to and reason about your decision.
3. Provide the treatment as a response after your reasoning steps.

List of treatments: {treatments}

{examples}

What treatment is this journal abstract about: {source}?
"""
