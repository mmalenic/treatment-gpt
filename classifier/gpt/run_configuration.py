from classifier.gpt.no_sources_classifier import NoSourcesGenePairGPTClassifier
from classifier.gpt.no_sources_no_list_classifier import (
    NoSourcesNoListGenePairGPTClassifier,
)
from classifier.gpt.prompt_templates import *
from classifier.gpt.treatment_only_classifier import TreatmentSourceGPTClassifier
from classifier.gpt.treatment_only_no_list_classifier import (
    TreatmentSourceNoListGPTClassifier,
)
from dataset.treatment_source_dataset import TreatmentSourceDataset
from dataset.gene_pair_dataset import GenePairDataset


class RunConfiguration:
    def __init__(
        self,
        gene_pair_dataset: GenePairDataset,
        gene_pair_dataset_no_list: GenePairDataset,
        treatment_source_dataset: TreatmentSourceDataset,
    ):
        """
        Initialize this class.
        """

        self._run_configuration = {
            "runs": [
                {
                    "run_name": Prompts.zero_shot_no_sources_no_list_name,
                    "model_type": "gpt-3.5-turbo",
                    "classifier": NoSourcesNoListGenePairGPTClassifier(
                        gene_pair_dataset_no_list,
                        Prompts.zero_shot_no_sources_no_list_name,
                        "gpt-3.5-turbo",
                        repeat_n_times=3,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.few_shot_no_sources_no_list_name,
                    "model_type": "gpt-3.5-turbo",
                    "classifier": NoSourcesNoListGenePairGPTClassifier(
                        gene_pair_dataset_no_list,
                        Prompts.few_shot_no_sources_no_list_name,
                        "gpt-3.5-turbo",
                        repeat_n_times=3,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.zero_shot_no_sources_no_list_cot_name,
                    "model_type": "gpt-3.5-turbo",
                    "classifier": NoSourcesNoListGenePairGPTClassifier(
                        gene_pair_dataset_no_list,
                        Prompts.zero_shot_no_sources_no_list_cot_name,
                        "gpt-3.5-turbo",
                        repeat_n_times=3,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.few_shot_no_sources_no_list_cot_name,
                    "model_type": "gpt-3.5-turbo",
                    "classifier": NoSourcesNoListGenePairGPTClassifier(
                        gene_pair_dataset_no_list,
                        Prompts.few_shot_no_sources_no_list_cot_name,
                        "gpt-3.5-turbo",
                        repeat_n_times=3,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.zero_shot_no_sources_name,
                    "model_type": "gpt-3.5-turbo",
                    "classifier": NoSourcesGenePairGPTClassifier(
                        gene_pair_dataset,
                        Prompts.zero_shot_no_sources_name,
                        "gpt-3.5-turbo",
                        repeat_n_times=3,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.few_shot_no_sources_name,
                    "model_type": "gpt-3.5-turbo",
                    "classifier": NoSourcesGenePairGPTClassifier(
                        gene_pair_dataset,
                        Prompts.few_shot_no_sources_name,
                        "gpt-3.5-turbo",
                        repeat_n_times=3,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.zero_shot_no_sources_cot_name,
                    "model_type": "gpt-3.5-turbo",
                    "classifier": NoSourcesGenePairGPTClassifier(
                        gene_pair_dataset,
                        Prompts.zero_shot_no_sources_cot_name,
                        "gpt-3.5-turbo",
                        repeat_n_times=3,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.few_shot_no_sources_cot_name,
                    "model_type": "gpt-3.5-turbo",
                    "classifier": NoSourcesGenePairGPTClassifier(
                        gene_pair_dataset,
                        Prompts.few_shot_no_sources_cot_name,
                        "gpt-3.5-turbo",
                        repeat_n_times=3,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.zero_shot_treatment_source_no_list_name,
                    "model_type": "gpt-3.5-turbo",
                    "classifier": TreatmentSourceNoListGPTClassifier(
                        treatment_source_dataset,
                        Prompts.zero_shot_treatment_source_no_list_name,
                        "gpt-3.5-turbo",
                        repeat_n_times=3,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.few_shot_treatment_source_no_list_name,
                    "model_type": "gpt-3.5-turbo",
                    "classifier": TreatmentSourceNoListGPTClassifier(
                        treatment_source_dataset,
                        Prompts.few_shot_treatment_source_no_list_name,
                        "gpt-3.5-turbo",
                        repeat_n_times=3,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.zero_shot_treatment_source_no_list_cot_name,
                    "model_type": "gpt-3.5-turbo",
                    "classifier": TreatmentSourceNoListGPTClassifier(
                        treatment_source_dataset,
                        Prompts.zero_shot_treatment_source_no_list_cot_name,
                        "gpt-3.5-turbo",
                        repeat_n_times=3,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.few_shot_treatment_source_no_list_cot_name,
                    "model_type": "gpt-3.5-turbo",
                    "classifier": TreatmentSourceNoListGPTClassifier(
                        treatment_source_dataset,
                        Prompts.few_shot_treatment_source_no_list_cot_name,
                        "gpt-3.5-turbo",
                        repeat_n_times=3,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.zero_shot_treatment_source_name,
                    "model_type": "gpt-3.5-turbo",
                    "classifier": TreatmentSourceGPTClassifier(
                        treatment_source_dataset,
                        Prompts.zero_shot_treatment_source_name,
                        "gpt-3.5-turbo",
                        repeat_n_times=3,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.few_shot_treatment_source_name,
                    "model_type": "gpt-3.5-turbo",
                    "classifier": TreatmentSourceGPTClassifier(
                        treatment_source_dataset,
                        Prompts.few_shot_treatment_source_name,
                        "gpt-3.5-turbo",
                        repeat_n_times=3,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.zero_shot_treatment_source_cot_name,
                    "model_type": "gpt-3.5-turbo",
                    "classifier": TreatmentSourceGPTClassifier(
                        treatment_source_dataset,
                        Prompts.zero_shot_treatment_source_cot_name,
                        "gpt-3.5-turbo",
                        repeat_n_times=3,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.few_shot_treatment_source_cot_name,
                    "model_type": "gpt-3.5-turbo",
                    "classifier": TreatmentSourceGPTClassifier(
                        treatment_source_dataset,
                        Prompts.few_shot_treatment_source_cot_name,
                        "gpt-3.5-turbo",
                        repeat_n_times=3,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.zero_shot_no_sources_no_list_name,
                    "model_type": "gpt-4",
                    "classifier": NoSourcesNoListGenePairGPTClassifier(
                        gene_pair_dataset_no_list,
                        Prompts.zero_shot_no_sources_no_list_name,
                        "gpt-4",
                        repeat_n_times=2,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.few_shot_no_sources_no_list_name,
                    "model_type": "gpt-4",
                    "classifier": NoSourcesNoListGenePairGPTClassifier(
                        gene_pair_dataset_no_list,
                        Prompts.few_shot_no_sources_no_list_name,
                        "gpt-4",
                        repeat_n_times=2,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.zero_shot_no_sources_no_list_cot_name,
                    "model_type": "gpt-4",
                    "classifier": NoSourcesNoListGenePairGPTClassifier(
                        gene_pair_dataset_no_list,
                        Prompts.zero_shot_no_sources_no_list_cot_name,
                        "gpt-4",
                        repeat_n_times=2,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.few_shot_no_sources_no_list_cot_name,
                    "model_type": "gpt-4",
                    "classifier": NoSourcesNoListGenePairGPTClassifier(
                        gene_pair_dataset_no_list,
                        Prompts.few_shot_no_sources_no_list_cot_name,
                        "gpt-4",
                        repeat_n_times=2,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.zero_shot_no_sources_name,
                    "model_type": "gpt-4",
                    "classifier": NoSourcesGenePairGPTClassifier(
                        gene_pair_dataset,
                        Prompts.zero_shot_no_sources_name,
                        "gpt-4",
                        repeat_n_times=2,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.few_shot_no_sources_name,
                    "model_type": "gpt-4",
                    "classifier": NoSourcesGenePairGPTClassifier(
                        gene_pair_dataset,
                        Prompts.few_shot_no_sources_name,
                        "gpt-4",
                        repeat_n_times=2,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.zero_shot_no_sources_cot_name,
                    "model_type": "gpt-4",
                    "classifier": NoSourcesGenePairGPTClassifier(
                        gene_pair_dataset,
                        Prompts.zero_shot_no_sources_cot_name,
                        "gpt-4",
                        repeat_n_times=2,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.few_shot_no_sources_cot_name,
                    "model_type": "gpt-4",
                    "classifier": NoSourcesGenePairGPTClassifier(
                        gene_pair_dataset,
                        Prompts.few_shot_no_sources_cot_name,
                        "gpt-4",
                        repeat_n_times=2,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.zero_shot_treatment_source_no_list_name,
                    "model_type": "gpt-4",
                    "classifier": TreatmentSourceNoListGPTClassifier(
                        treatment_source_dataset,
                        Prompts.zero_shot_treatment_source_no_list_name,
                        "gpt-4",
                        repeat_n_times=2,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.few_shot_treatment_source_no_list_name,
                    "model_type": "gpt-4",
                    "classifier": TreatmentSourceNoListGPTClassifier(
                        treatment_source_dataset,
                        Prompts.few_shot_treatment_source_no_list_name,
                        "gpt-4",
                        repeat_n_times=2,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.zero_shot_treatment_source_no_list_cot_name,
                    "model_type": "gpt-4",
                    "classifier": TreatmentSourceNoListGPTClassifier(
                        treatment_source_dataset,
                        Prompts.zero_shot_treatment_source_no_list_cot_name,
                        "gpt-4",
                        repeat_n_times=2,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.few_shot_treatment_source_no_list_cot_name,
                    "model_type": "gpt-4",
                    "classifier": TreatmentSourceNoListGPTClassifier(
                        treatment_source_dataset,
                        Prompts.few_shot_treatment_source_no_list_cot_name,
                        "gpt-4",
                        repeat_n_times=2,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.zero_shot_treatment_source_name,
                    "model_type": "gpt-4",
                    "classifier": TreatmentSourceGPTClassifier(
                        treatment_source_dataset,
                        Prompts.zero_shot_treatment_source_name,
                        "gpt-4",
                        repeat_n_times=2,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.few_shot_treatment_source_name,
                    "model_type": "gpt-4",
                    "classifier": TreatmentSourceGPTClassifier(
                        treatment_source_dataset,
                        Prompts.few_shot_treatment_source_name,
                        "gpt-4",
                        repeat_n_times=2,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.zero_shot_treatment_source_cot_name,
                    "model_type": "gpt-4",
                    "classifier": TreatmentSourceGPTClassifier(
                        treatment_source_dataset,
                        Prompts.zero_shot_treatment_source_cot_name,
                        "gpt-4",
                        repeat_n_times=2,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.few_shot_treatment_source_cot_name,
                    "model_type": "gpt-4",
                    "classifier": TreatmentSourceGPTClassifier(
                        treatment_source_dataset,
                        Prompts.few_shot_treatment_source_cot_name,
                        "gpt-4",
                        repeat_n_times=2,
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
            ]
        }

    @property
    def run_configuration(self):
        """
        Get the run configuration
        """
        return self._run_configuration

    def calculate_costs(self):
        """
        Calculate the cost of each run.
        """
        for run in self.run_configuration["runs"]:
            run["cost_estimate"] = run["classifier"].cost_estimate
            run["max_tokens"] = run["classifier"].max_token_number

    def max_tokens(self):
        """
        The maximum amount of tokens across all runs.
        """
        return max([x["max_tokens"] for x in self.run_configuration["runs"]])

    def total_cost(self):
        """
        Calculate the total cost
        """
        return sum([x["cost_estimate"] for x in self.run_configuration["runs"]])

    def save_example_prompts(self):
        """
        Save example prompts.
        """
        for run in self.run_configuration["runs"]:
            run["classifier"].save_example_prompt()
