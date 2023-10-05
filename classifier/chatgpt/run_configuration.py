from classifier.chatgpt.no_sources_classifier import NoSourcesGenePairGPTClassifier
from classifier.chatgpt.prompt_templates import *
from classifier.chatgpt.treatment_only_classifier import TreatmentSourceGPTClassifier
from classifier.chatgpt.with_sources_classifier import WithSourcesGenePairGPTClassifier
from dataset.treatment_source_dataset import TreatmentSourceDataset
from dataset.gene_pair_dataset import GenePairDataset


class RunConfiguration:
    def __init__(
        self,
        gene_pair_dataset: GenePairDataset,
        treatment_source_dataset: TreatmentSourceDataset,
    ):
        """
        Initialize this class.
        """

        self._run_configuration = {
            "runs": [
                {
                    "run_name": Prompts.zero_shot_no_sources_name,
                    "model_type": "gpt-3.5-turbo",
                    "classifier": NoSourcesGenePairGPTClassifier(
                        gene_pair_dataset,
                        Prompts.zero_shot_no_sources_name,
                        "gpt-3.5-turbo",
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
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.zero_shot_with_sources_name,
                    "model_type": "gpt-3.5-turbo",
                    "classifier": WithSourcesGenePairGPTClassifier(
                        gene_pair_dataset,
                        Prompts.zero_shot_with_sources_name,
                        "gpt-3.5-turbo",
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.few_shot_with_sources_name,
                    "model_type": "gpt-3.5-turbo",
                    "classifier": WithSourcesGenePairGPTClassifier(
                        gene_pair_dataset,
                        Prompts.few_shot_with_sources_name,
                        "gpt-3.5-turbo",
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.zero_shot_with_sources_cot_name,
                    "model_type": "gpt-3.5-turbo",
                    "classifier": WithSourcesGenePairGPTClassifier(
                        gene_pair_dataset,
                        Prompts.zero_shot_with_sources_cot_name,
                        "gpt-3.5-turbo",
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.few_shot_with_sources_cot_name,
                    "model_type": "gpt-3.5-turbo",
                    "classifier": WithSourcesGenePairGPTClassifier(
                        gene_pair_dataset,
                        Prompts.few_shot_with_sources_cot_name,
                        "gpt-3.5-turbo",
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
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.zero_shot_no_sources_name,
                    "model_type": "gpt-4",
                    "classifier": NoSourcesGenePairGPTClassifier(
                        gene_pair_dataset, Prompts.zero_shot_no_sources_name, "gpt-4"
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.few_shot_no_sources_name,
                    "model_type": "gpt-4",
                    "classifier": NoSourcesGenePairGPTClassifier(
                        gene_pair_dataset, Prompts.few_shot_no_sources_name, "gpt-4"
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
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.few_shot_no_sources_cot_name,
                    "model_type": "gpt-4",
                    "classifier": NoSourcesGenePairGPTClassifier(
                        gene_pair_dataset, Prompts.few_shot_no_sources_cot_name, "gpt-4"
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.zero_shot_with_sources_name,
                    "model_type": "gpt-4",
                    "classifier": WithSourcesGenePairGPTClassifier(
                        gene_pair_dataset, Prompts.zero_shot_with_sources_name, "gpt-4"
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.few_shot_with_sources_name,
                    "model_type": "gpt-4",
                    "classifier": WithSourcesGenePairGPTClassifier(
                        gene_pair_dataset, Prompts.few_shot_with_sources_name, "gpt-4"
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.zero_shot_with_sources_cot_name,
                    "model_type": "gpt-4",
                    "classifier": WithSourcesGenePairGPTClassifier(
                        gene_pair_dataset,
                        Prompts.zero_shot_with_sources_cot_name,
                        "gpt-4",
                    ),
                    "cost_estimate": None,
                    "max_tokens": None,
                },
                {
                    "run_name": Prompts.few_shot_with_sources_cot_name,
                    "model_type": "gpt-4",
                    "classifier": WithSourcesGenePairGPTClassifier(
                        gene_pair_dataset,
                        Prompts.few_shot_with_sources_cot_name,
                        "gpt-4",
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

    def total_cost(self):
        """
        Calculate the total cost
        """
        return sum([x["cost_estimate"] for x in self.run_configuration["runs"]])
