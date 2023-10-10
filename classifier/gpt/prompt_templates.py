from inspect import cleandoc
from textwrap import dedent
from typing import Literal


def const_literals(cls):
    for const, lit in dict(**vars(cls)).items():
        setattr(cls, f"{const}_type", Literal[lit])
        setattr(cls, f"{const}_literal", Literal[const])
        setattr(cls, f"{const}_name", const)

    return cls


@const_literals
class Prompts:
    @staticmethod
    def from_name(name: str) -> str:
        return getattr(Prompts, name)

    _multi_label_json_task_prompt_template = "Provide your response in a JSON format containing a single key `treatments` and a value corresponding to the array of assigned treatments."

    _single_label_json_task_prompt_template = "Provide your response in a JSON format containing a single key `treatment` and a value corresponding to the treatment."

    _json_response_prompt_template = "Your JSON response:"

    _no_additional_information_prompt_template = (
        "Do no provide any additional information."
    )

    _patient_treatment_task_prompt_template = """You are tasked with classifying a cancer patient into treatments.
You will have the following information available:
1. The patient's cancer type.
2. A pair of the patient's actionable genes."""

    _patient_treatment_task_treatments_prompt_template = (
        "The list of possible treatments, delimited with square brackets."
    )

    _patient_treatment_source_task_prompt_template = """You are tasked with classifying a cancer patient into treatments.
You will have the following information available:
1. The pairs of possible treatments and abstracts outlining the source of the treatment.
2. The patient's cancer type.
3. A pair of the patient's actionable genes."""

    _multi_label_task_prompt_template = "Assign this patient to {n_treatments} treatments based on probabilities. Not all treatments are correct."

    _multi_label_task_no_list_prompt_template = "Assign this patient to {n_treatments} treatments based on probabilities. Choose specific drug names or clinical trials and avoid generic terms like `chemotherapy`. Use a `+` to denote combination therapy if applicable."

    _list_of_treatments_prompt_template = "List of possible treatments: {treatments}"

    _list_of_treatments_only_prompt_template = "List of all treatments: {treatments}"

    _list_of_treatments_and_sources_prompt_template = """List of possible treatments and sources:
{treatments_and_sources}"""

    _patient_cancer_type_prompt_template = """A patient has {cancer_type} with the following actionable genes: {genes}.
What treatments are available for this patient?"""

    _few_shot_examples_prompt_template = "A few prior examples of classifications. The examples are delimited with triple backticks."

    _examples_prompt_template = """Examples:
{examples}"""

    cot_prompt_template = "Think step by step about which treatment is best and reason about your decision."

    _treatment_only_task_prompt_template = """You are tasked with classifying which treatment a journal abstract is about.
You will have the following information available:
1. The list of all possible treatments, delimited with square brackets.
2. The journal abstract."""

    _treatment_only_classification_task_prompt_template = (
        "Classify the journal abstract into a treatment. Only one treatment is correct."
    )

    _treatment_only_response_prompt_template = """Journal abstract:
{source}
What treatment is this journal abstract about?"""

    treatment_only_cot_prompt_template = "Think step by step about which treatment the abstract refers to and reason about your decision."

    gene_pair_no_list_example_prompt_template = f"""```
{_patient_cancer_type_prompt_template}
{{answer}}
```
"""

    gene_pair_example_prompt_template = f"""```
{_list_of_treatments_prompt_template}

{_patient_cancer_type_prompt_template}
{{answer}}
```
"""

    gene_pair_sources_example_prompt_template = f"""```
{_list_of_treatments_and_sources_prompt_template}
{_patient_cancer_type_prompt_template}
{{answer}}
```
"""

    treatment_source_prompt_template = f"""```
{_treatment_only_response_prompt_template}
{{answer}}
```
"""

    treatment_and_source_prompt_template = """'''
Treatment: {treatment}
Source: {source}'''
"""

    zero_shot_no_sources_no_list = f"""{_patient_treatment_task_prompt_template}

Perform the following tasks:
1. {_multi_label_task_no_list_prompt_template}
2. {_multi_label_json_task_prompt_template} {_no_additional_information_prompt_template}

{_patient_cancer_type_prompt_template}
{_json_response_prompt_template}
"""

    few_shot_no_sources_no_list = f"""{_patient_treatment_task_prompt_template}
3. {_few_shot_examples_prompt_template}

Perform the following tasks:
1. {_multi_label_task_no_list_prompt_template}
2. {_multi_label_json_task_prompt_template} {_no_additional_information_prompt_template}

{_examples_prompt_template}

{_patient_cancer_type_prompt_template}
{_json_response_prompt_template}
"""

    zero_shot_no_sources_no_list_cot = f"""{_patient_treatment_task_prompt_template}

Perform the following tasks:
1. {_multi_label_task_no_list_prompt_template}
2. {cot_prompt_template}
3. {_multi_label_json_task_prompt_template}

{_patient_cancer_type_prompt_template}
"""

    few_shot_no_sources_no_list_cot = f"""{_patient_treatment_task_prompt_template}
3. {_few_shot_examples_prompt_template}

Perform the following tasks:
1. {_multi_label_task_no_list_prompt_template}
2. {cot_prompt_template}
3. {_multi_label_json_task_prompt_template}

{_examples_prompt_template}

{_patient_cancer_type_prompt_template}
"""

    zero_shot_no_sources = f"""{_patient_treatment_task_prompt_template}
3. {_patient_treatment_task_treatments_prompt_template}

Perform the following tasks:
1. {_multi_label_task_prompt_template}
2. {_multi_label_json_task_prompt_template} {_no_additional_information_prompt_template}

{_list_of_treatments_prompt_template}

{_patient_cancer_type_prompt_template}
{_json_response_prompt_template}
"""

    few_shot_no_sources = f"""{_patient_treatment_task_prompt_template}
3. {_patient_treatment_task_treatments_prompt_template}
4. {_few_shot_examples_prompt_template}

Perform the following tasks:
1. {_multi_label_task_prompt_template}
2. {_multi_label_json_task_prompt_template} {_no_additional_information_prompt_template}

{_examples_prompt_template}
{_list_of_treatments_prompt_template}

{_patient_cancer_type_prompt_template}
{_json_response_prompt_template}
"""

    zero_shot_no_sources_cot = f"""{_patient_treatment_task_prompt_template}
3. {_patient_treatment_task_treatments_prompt_template}

Perform the following tasks:
1. {_multi_label_task_prompt_template}
2. {cot_prompt_template}
3. {_multi_label_json_task_prompt_template}

{_list_of_treatments_prompt_template}

{_patient_cancer_type_prompt_template}
"""

    few_shot_no_sources_cot = f"""{_patient_treatment_task_prompt_template}
3. {_patient_treatment_task_treatments_prompt_template}
4. {_few_shot_examples_prompt_template}

Perform the following tasks:
1. {_multi_label_task_prompt_template}
2. {cot_prompt_template}
3. {_multi_label_json_task_prompt_template}

{_examples_prompt_template}
{_list_of_treatments_prompt_template}

{_patient_cancer_type_prompt_template}
"""

    zero_shot_with_sources = f"""{_patient_treatment_source_task_prompt_template}

Perform the following tasks:
1. {_multi_label_task_prompt_template}
2. {_multi_label_json_task_prompt_template} {_no_additional_information_prompt_template}

{_list_of_treatments_and_sources_prompt_template}
{_patient_cancer_type_prompt_template}
{_json_response_prompt_template}
"""

    few_shot_with_sources = f"""{_patient_treatment_source_task_prompt_template}
4. {_few_shot_examples_prompt_template}

Perform the following tasks:
1. {_multi_label_task_prompt_template}
2. {_multi_label_json_task_prompt_template} {_no_additional_information_prompt_template}

{_examples_prompt_template}
{_list_of_treatments_and_sources_prompt_template}
{_patient_cancer_type_prompt_template}
{_json_response_prompt_template}
"""

    zero_shot_with_sources_cot = f"""{_patient_treatment_source_task_prompt_template}

Perform the following tasks:
1. {_multi_label_task_prompt_template}
2. {cot_prompt_template}
3. {_multi_label_json_task_prompt_template}

{_list_of_treatments_and_sources_prompt_template}
{_patient_cancer_type_prompt_template}
"""

    few_shot_with_sources_cot = f"""{_patient_treatment_source_task_prompt_template}
4. {_few_shot_examples_prompt_template}

Perform the following tasks:
1. {_multi_label_task_prompt_template}
2. {cot_prompt_template}
3. {_multi_label_json_task_prompt_template}

{_examples_prompt_template}
{_list_of_treatments_and_sources_prompt_template}
{_patient_cancer_type_prompt_template}
"""

    zero_shot_treatment_source = f"""{_treatment_only_task_prompt_template}

Perform the following tasks:
1. {_treatment_only_classification_task_prompt_template}
2. {_single_label_json_task_prompt_template} {_no_additional_information_prompt_template}

{_list_of_treatments_only_prompt_template}

{_treatment_only_response_prompt_template}
{_json_response_prompt_template}
"""

    few_shot_treatment_source = f"""{_treatment_only_task_prompt_template}
3. {_few_shot_examples_prompt_template}

Perform the following tasks:
1. {_treatment_only_classification_task_prompt_template}
2. {_single_label_json_task_prompt_template} {_no_additional_information_prompt_template}

{_list_of_treatments_only_prompt_template}

{_examples_prompt_template}
{_treatment_only_response_prompt_template}
{_json_response_prompt_template}
"""

    zero_shot_treatment_source_cot = f"""{_treatment_only_task_prompt_template}

Perform the following tasks:
1. {_treatment_only_classification_task_prompt_template}
2. {treatment_only_cot_prompt_template}
3. {_single_label_json_task_prompt_template}

{_list_of_treatments_only_prompt_template}

{_treatment_only_response_prompt_template}
"""

    few_shot_treatment_source_cot = f"""{_treatment_only_task_prompt_template}
3. {_few_shot_examples_prompt_template}

Perform the following tasks:
1. {_treatment_only_classification_task_prompt_template}
2. {treatment_only_cot_prompt_template}
3. {_single_label_json_task_prompt_template}

{_list_of_treatments_only_prompt_template}

{_examples_prompt_template}
{_treatment_only_response_prompt_template}
"""
