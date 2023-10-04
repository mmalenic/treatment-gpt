_MULTI_LABEL_JSON_TASK_PROMPT_TEMPLATE = "Provide your response in a JSON format containing a single key `treatments` and a value corresponding to the array of assigned treatments."

_SINGLE_LABEL_JSON_TASK_PROMPT_TEMPLATE = "Provide your response in a JSON format containing a single key `treatment` and a value corresponding to the treatment."

_JSON_RESPONSE_PROMPT_TEMPLATE = "Your JSON response:"

_NO_ADDITIONAL_INFORMATION_PROMPT_TEMPLATE = "Do no provide any additional information."

_PATIENT_TREATMENT_TASK_PROMPT_TEMPLATE = """You are tasked with classifying a cancer patient into treatments.
You will have the following information available:
1. The list of possible treatments, delimited with square brackets. Not all treatments are correct.
2. The patient's cancer type.
3. A pair of the patient's actionable genes."""

_PATIENT_TREATMENT_SOURCE_TASK_PROMPT_TEMPLATE = """You are tasked with classifying a cancer patient into treatments.
You will have the following information available:
1. The list of pairs of possible treatments and abstracts outlining the source of the treatment. Each treatment and source pair is delimited by triple quotes.
2. The patient's cancer type.
3. A pair of the patient's actionable genes."""

_MULTI_LABEL_TASK_PROMPT_TEMPLATE = "Assign this patient to {n_treatments} treatments based on probabilities. Not all treatments are correct."

_LIST_OF_TREATMENTS_PROMPT_TEMPLATE = "List of possible treatments: {treatments}"

_LIST_OF_TREATMENTS_ONLY_PROMPT_TEMPLATE = "List of all treatments: {treatments}"

_LIST_OF_TREATMENTS_AND_SOURCES_PROMPT_TEMPLATE = (
    "List of possible treatments and sources: {treatments_and_sources}"
)

_PATIENT_CANCER_TYPE_PROMPT_TEMPLATE = """A patient has {cancer_type} with the following actionable genes: {genes}.
What treatments are available for this patient?"""

_FEW_SHOT_EXAMPLES_PROMPT_TEMPLATE = "A few prior examples of classifications. The examples are delimited with triple backticks."

_EXAMPLES_PROMPT_TEMPLATE = """Examples:
{examples}"""

_COT_PROMPT_TEMPLATE = (
    "Think step by step about which treatment is best and reason about your decision."
)

_TREATMENT_ONLY_TASK_PROMPT_TEMPLATE = """You are tasked with classifying which treatment a journal abstract is about.
You will have the following information available:
1. The list of all possible treatments, delimited with square brackets.
2. The journal abstract."""

_TREATMENT_ONLY_CLASSIFICATION_TASK_PROMPT_TEMPLATE = (
    "Classify the journal abstract into a treatment. Only one treatment is correct."
)

_TREATMENT_ONLY_RESPONSE_PROMPT_TEMPLATE = """Journal abstract:
{source}
What treatment is this journal abstract about?"""

_TREATMENT_ONLY_COT_PROMPT_TEMPLATE = "Think step by step about which treatment the abstract refers to and reason about your decision."

GENE_PAIR_EXAMPLE_PROMPT_TEMPLATE = f"""```
{_LIST_OF_TREATMENTS_PROMPT_TEMPLATE}

{_PATIENT_CANCER_TYPE_PROMPT_TEMPLATE}
{{answer}}
```
"""

GENE_PAIR_SOURCES_EXAMPLE_PROMPT_TEMPLATE = f"""```
{_LIST_OF_TREATMENTS_AND_SOURCES_PROMPT_TEMPLATE}

{_PATIENT_CANCER_TYPE_PROMPT_TEMPLATE}
{{answer}}
```
"""

TREATMENT_SOURCE_PROMPT_TEMPLATE = f"""```
{_TREATMENT_ONLY_RESPONSE_PROMPT_TEMPLATE}
{{answer}}
```
"""

TREATMENT_AND_SOURCE_PROMPT_TEMPLATE = """

'''
Treatment: {treatment}
Source: {source}
'''
"""

ZERO_SHOT_PROMPT_TEMPLATE = f"""{_PATIENT_TREATMENT_TASK_PROMPT_TEMPLATE}

Perform the following tasks:
1. {_MULTI_LABEL_TASK_PROMPT_TEMPLATE}
2. {_MULTI_LABEL_JSON_TASK_PROMPT_TEMPLATE} {_NO_ADDITIONAL_INFORMATION_PROMPT_TEMPLATE}

{_LIST_OF_TREATMENTS_PROMPT_TEMPLATE}

{_PATIENT_CANCER_TYPE_PROMPT_TEMPLATE}
{_JSON_RESPONSE_PROMPT_TEMPLATE}
"""

FEW_SHOT_PROMPT_TEMPLATE = f"""{_PATIENT_TREATMENT_TASK_PROMPT_TEMPLATE}
4. {_FEW_SHOT_EXAMPLES_PROMPT_TEMPLATE}

Perform the following tasks:
1. {_MULTI_LABEL_TASK_PROMPT_TEMPLATE}
2. {_MULTI_LABEL_JSON_TASK_PROMPT_TEMPLATE} {_NO_ADDITIONAL_INFORMATION_PROMPT_TEMPLATE}

{_EXAMPLES_PROMPT_TEMPLATE}
{_LIST_OF_TREATMENTS_PROMPT_TEMPLATE}

{_PATIENT_CANCER_TYPE_PROMPT_TEMPLATE}
{_JSON_RESPONSE_PROMPT_TEMPLATE}
"""

ZERO_SHOT_COT_PROMPT_TEMPLATE = f"""{_PATIENT_TREATMENT_TASK_PROMPT_TEMPLATE}

Perform the following tasks:
1. {_MULTI_LABEL_TASK_PROMPT_TEMPLATE}
2. {_COT_PROMPT_TEMPLATE}
3. {_MULTI_LABEL_JSON_TASK_PROMPT_TEMPLATE}

{_LIST_OF_TREATMENTS_PROMPT_TEMPLATE}

{_PATIENT_CANCER_TYPE_PROMPT_TEMPLATE}
"""

FEW_SHOT_COT_PROMPT_TEMPLATE = f"""{_PATIENT_TREATMENT_TASK_PROMPT_TEMPLATE}
4. {_FEW_SHOT_EXAMPLES_PROMPT_TEMPLATE}

Perform the following tasks:
1. {_MULTI_LABEL_TASK_PROMPT_TEMPLATE}
2. {_COT_PROMPT_TEMPLATE}
3. {_MULTI_LABEL_JSON_TASK_PROMPT_TEMPLATE}

{_EXAMPLES_PROMPT_TEMPLATE}
{_LIST_OF_TREATMENTS_PROMPT_TEMPLATE}

{_PATIENT_CANCER_TYPE_PROMPT_TEMPLATE}
"""

ZERO_SHOT_WITH_SOURCES_PROMPT_TEMPLATE = f"""{_PATIENT_TREATMENT_SOURCE_TASK_PROMPT_TEMPLATE}

Perform the following tasks:
1. {_MULTI_LABEL_TASK_PROMPT_TEMPLATE}
2. {_MULTI_LABEL_JSON_TASK_PROMPT_TEMPLATE} {_NO_ADDITIONAL_INFORMATION_PROMPT_TEMPLATE}

{_LIST_OF_TREATMENTS_AND_SOURCES_PROMPT_TEMPLATE}

{_PATIENT_CANCER_TYPE_PROMPT_TEMPLATE}
{_JSON_RESPONSE_PROMPT_TEMPLATE}
"""

FEW_SHOT_WITH_SOURCES_PROMPT_TEMPLATE = f"""{_PATIENT_TREATMENT_SOURCE_TASK_PROMPT_TEMPLATE}
4. {_FEW_SHOT_EXAMPLES_PROMPT_TEMPLATE}

Perform the following tasks:
1. {_MULTI_LABEL_TASK_PROMPT_TEMPLATE}
2. {_MULTI_LABEL_JSON_TASK_PROMPT_TEMPLATE} {_NO_ADDITIONAL_INFORMATION_PROMPT_TEMPLATE}

{_EXAMPLES_PROMPT_TEMPLATE}
{_LIST_OF_TREATMENTS_AND_SOURCES_PROMPT_TEMPLATE}

{_PATIENT_CANCER_TYPE_PROMPT_TEMPLATE}
{_JSON_RESPONSE_PROMPT_TEMPLATE}
"""

ZERO_SHOT_COT_WITH_SOURCES_PROMPT_TEMPLATE = f"""{_PATIENT_TREATMENT_SOURCE_TASK_PROMPT_TEMPLATE}

Perform the following tasks:
1. {_MULTI_LABEL_TASK_PROMPT_TEMPLATE}
2. {_COT_PROMPT_TEMPLATE}
3. {_MULTI_LABEL_JSON_TASK_PROMPT_TEMPLATE}

{_LIST_OF_TREATMENTS_AND_SOURCES_PROMPT_TEMPLATE}

{_PATIENT_CANCER_TYPE_PROMPT_TEMPLATE}
"""

FEW_SHOT_COT_WITH_SOURCES_PROMPT_TEMPLATE = f"""{_PATIENT_TREATMENT_SOURCE_TASK_PROMPT_TEMPLATE}

4. {_FEW_SHOT_EXAMPLES_PROMPT_TEMPLATE}

Perform the following tasks:
1. {_MULTI_LABEL_TASK_PROMPT_TEMPLATE}
2. {_COT_PROMPT_TEMPLATE}
3. {_MULTI_LABEL_JSON_TASK_PROMPT_TEMPLATE}

{_EXAMPLES_PROMPT_TEMPLATE}
{_LIST_OF_TREATMENTS_AND_SOURCES_PROMPT_TEMPLATE}

{_PATIENT_CANCER_TYPE_PROMPT_TEMPLATE}
"""

ZERO_SHOT_TREATMENT_ONLY_PROMPT_TEMPLATE = f"""{_TREATMENT_ONLY_TASK_PROMPT_TEMPLATE}

Perform the following tasks:
1. {_TREATMENT_ONLY_CLASSIFICATION_TASK_PROMPT_TEMPLATE}
2. {_SINGLE_LABEL_JSON_TASK_PROMPT_TEMPLATE} {_NO_ADDITIONAL_INFORMATION_PROMPT_TEMPLATE}

{_LIST_OF_TREATMENTS_ONLY_PROMPT_TEMPLATE}

{_TREATMENT_ONLY_RESPONSE_PROMPT_TEMPLATE}
{_JSON_RESPONSE_PROMPT_TEMPLATE}
"""

FEW_SHOT_TREATMENT_ONLY_PROMPT_TEMPLATE = f"""{_TREATMENT_ONLY_TASK_PROMPT_TEMPLATE}
3. {_FEW_SHOT_EXAMPLES_PROMPT_TEMPLATE}

Perform the following tasks:
1. {_TREATMENT_ONLY_CLASSIFICATION_TASK_PROMPT_TEMPLATE}
2. {_SINGLE_LABEL_JSON_TASK_PROMPT_TEMPLATE} {_NO_ADDITIONAL_INFORMATION_PROMPT_TEMPLATE}

{_LIST_OF_TREATMENTS_ONLY_PROMPT_TEMPLATE}

{_EXAMPLES_PROMPT_TEMPLATE}
{_TREATMENT_ONLY_RESPONSE_PROMPT_TEMPLATE}
{_JSON_RESPONSE_PROMPT_TEMPLATE}
"""

ZERO_SHOT_TREATMENT_ONLY_COT_PROMPT_TEMPLATE = f"""{_TREATMENT_ONLY_TASK_PROMPT_TEMPLATE}

Perform the following tasks:
1. {_TREATMENT_ONLY_CLASSIFICATION_TASK_PROMPT_TEMPLATE}
2. {_TREATMENT_ONLY_COT_PROMPT_TEMPLATE}
3. {_SINGLE_LABEL_JSON_TASK_PROMPT_TEMPLATE}

{_LIST_OF_TREATMENTS_ONLY_PROMPT_TEMPLATE}

{_TREATMENT_ONLY_RESPONSE_PROMPT_TEMPLATE}
"""

FEW_SHOT_TREATMENT_ONLY_COT_PROMPT_TEMPLATE = f"""{_TREATMENT_ONLY_TASK_PROMPT_TEMPLATE}
3. {_FEW_SHOT_EXAMPLES_PROMPT_TEMPLATE}

Perform the following tasks:
1. {_TREATMENT_ONLY_CLASSIFICATION_TASK_PROMPT_TEMPLATE}
2. {_TREATMENT_ONLY_COT_PROMPT_TEMPLATE}
3. {_SINGLE_LABEL_JSON_TASK_PROMPT_TEMPLATE}

{_LIST_OF_TREATMENTS_ONLY_PROMPT_TEMPLATE}

{_EXAMPLES_PROMPT_TEMPLATE}
{_TREATMENT_ONLY_RESPONSE_PROMPT_TEMPLATE}
"""
