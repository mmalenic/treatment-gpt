import pandas as pd
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score

import classifier
from classifier import util
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

from sklearn.dummy import DummyClassifier


class RunConfiguration:
    def __init__(
        self,
        gene_pair_dataset: GenePairDataset,
        treatment_source_dataset: TreatmentSourceDataset,
    ):
        """
        Initialize this class.
        """

        self._gene_pair_dataset = gene_pair_dataset
        self._treatment_source_dataset = treatment_source_dataset

        self._treatment_source_results = pd.DataFrame()
        self._gene_pair_results = pd.DataFrame()

        self._run_configuration = {
            "runs": [
                # {
                #     "run_name": Prompts.zero_shot_no_sources_no_list_name,
                #     "model_type": "gpt-3.5-turbo",
                #     "classifier": NoSourcesNoListGenePairGPTClassifier(
                #         gene_pair_dataset,
                #         Prompts.zero_shot_no_sources_no_list_name,
                #         "gpt-3.5-turbo",
                #         repeat_n_times=3,
                #     ),
                #     "cost_estimate": None,
                #     "max_tokens": None,
                # },
                # {
                #     "run_name": Prompts.few_shot_no_sources_no_list_name,
                #     "model_type": "gpt-3.5-turbo",
                #     "classifier": NoSourcesNoListGenePairGPTClassifier(
                #         gene_pair_dataset,
                #         Prompts.few_shot_no_sources_no_list_name,
                #         "gpt-3.5-turbo",
                #         repeat_n_times=3,
                #     ),
                #     "cost_estimate": None,
                #     "max_tokens": None,
                # },
                # {
                #     "run_name": Prompts.zero_shot_no_sources_no_list_cot_name,
                #     "model_type": "gpt-3.5-turbo",
                #     "classifier": NoSourcesNoListGenePairGPTClassifier(
                #         gene_pair_dataset,
                #         Prompts.zero_shot_no_sources_no_list_cot_name,
                #         "gpt-3.5-turbo",
                #         repeat_n_times=3,
                #     ),
                #     "cost_estimate": None,
                #     "max_tokens": None,
                # },
                # {
                #     "run_name": Prompts.few_shot_no_sources_no_list_cot_name,
                #     "model_type": "gpt-3.5-turbo",
                #     "classifier": NoSourcesNoListGenePairGPTClassifier(
                #         gene_pair_dataset,
                #         Prompts.few_shot_no_sources_no_list_cot_name,
                #         "gpt-3.5-turbo",
                #         repeat_n_times=3,
                #     ),
                #     "cost_estimate": None,
                #     "max_tokens": None,
                # },
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
                # {
                #     "run_name": Prompts.few_shot_no_sources_name,
                #     "model_type": "gpt-3.5-turbo",
                #     "classifier": NoSourcesGenePairGPTClassifier(
                #         gene_pair_dataset,
                #         Prompts.few_shot_no_sources_name,
                #         "gpt-3.5-turbo",
                #         repeat_n_times=3,
                #     ),
                #     "cost_estimate": None,
                #     "max_tokens": None,
                # },
                # {
                #     "run_name": Prompts.zero_shot_no_sources_cot_name,
                #     "model_type": "gpt-3.5-turbo",
                #     "classifier": NoSourcesGenePairGPTClassifier(
                #         gene_pair_dataset,
                #         Prompts.zero_shot_no_sources_cot_name,
                #         "gpt-3.5-turbo",
                #         repeat_n_times=3,
                #     ),
                #     "cost_estimate": None,
                #     "max_tokens": None,
                # },
                # {
                #     "run_name": Prompts.few_shot_no_sources_cot_name,
                #     "model_type": "gpt-3.5-turbo",
                #     "classifier": NoSourcesGenePairGPTClassifier(
                #         gene_pair_dataset,
                #         Prompts.few_shot_no_sources_cot_name,
                #         "gpt-3.5-turbo",
                #         repeat_n_times=3,
                #     ),
                #     "cost_estimate": None,
                #     "max_tokens": None,
                # },
                # {
                #     "run_name": Prompts.zero_shot_treatment_source_no_list_name,
                #     "model_type": "gpt-3.5-turbo",
                #     "classifier": TreatmentSourceNoListGPTClassifier(
                #         treatment_source_dataset,
                #         Prompts.zero_shot_treatment_source_no_list_name,
                #         "gpt-3.5-turbo",
                #         repeat_n_times=3,
                #     ),
                #     "cost_estimate": None,
                #     "max_tokens": None,
                # },
                # {
                #     "run_name": Prompts.few_shot_treatment_source_no_list_name,
                #     "model_type": "gpt-3.5-turbo",
                #     "classifier": TreatmentSourceNoListGPTClassifier(
                #         treatment_source_dataset,
                #         Prompts.few_shot_treatment_source_no_list_name,
                #         "gpt-3.5-turbo",
                #         repeat_n_times=3,
                #     ),
                #     "cost_estimate": None,
                #     "max_tokens": None,
                # },
                # {
                #     "run_name": Prompts.zero_shot_treatment_source_no_list_cot_name,
                #     "model_type": "gpt-3.5-turbo",
                #     "classifier": TreatmentSourceNoListGPTClassifier(
                #         treatment_source_dataset,
                #         Prompts.zero_shot_treatment_source_no_list_cot_name,
                #         "gpt-3.5-turbo",
                #         repeat_n_times=3,
                #     ),
                #     "cost_estimate": None,
                #     "max_tokens": None,
                # },
                # {
                #     "run_name": Prompts.few_shot_treatment_source_no_list_cot_name,
                #     "model_type": "gpt-3.5-turbo",
                #     "classifier": TreatmentSourceNoListGPTClassifier(
                #         treatment_source_dataset,
                #         Prompts.few_shot_treatment_source_no_list_cot_name,
                #         "gpt-3.5-turbo",
                #         repeat_n_times=3,
                #     ),
                #     "cost_estimate": None,
                #     "max_tokens": None,
                # },
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
                # {
                #     "run_name": Prompts.few_shot_treatment_source_name,
                #     "model_type": "gpt-3.5-turbo",
                #     "classifier": TreatmentSourceGPTClassifier(
                #         treatment_source_dataset,
                #         Prompts.few_shot_treatment_source_name,
                #         "gpt-3.5-turbo",
                #         repeat_n_times=3,
                #     ),
                #     "cost_estimate": None,
                #     "max_tokens": None,
                # },
                # {
                #     "run_name": Prompts.zero_shot_treatment_source_cot_name,
                #     "model_type": "gpt-3.5-turbo",
                #     "classifier": TreatmentSourceGPTClassifier(
                #         treatment_source_dataset,
                #         Prompts.zero_shot_treatment_source_cot_name,
                #         "gpt-3.5-turbo",
                #         repeat_n_times=3,
                #     ),
                #     "cost_estimate": None,
                #     "max_tokens": None,
                # },
                # {
                #     "run_name": Prompts.few_shot_treatment_source_cot_name,
                #     "model_type": "gpt-3.5-turbo",
                #     "classifier": TreatmentSourceGPTClassifier(
                #         treatment_source_dataset,
                #         Prompts.few_shot_treatment_source_cot_name,
                #         "gpt-3.5-turbo",
                #         repeat_n_times=3,
                #     ),
                #     "cost_estimate": None,
                #     "max_tokens": None,
                # },
                # {
                #     "run_name": Prompts.zero_shot_no_sources_no_list_name,
                #     "model_type": "gpt-4",
                #     "classifier": NoSourcesNoListGenePairGPTClassifier(
                #         gene_pair_dataset,
                #         Prompts.zero_shot_no_sources_no_list_name,
                #         "gpt-4",
                #         repeat_n_times=2,
                #     ),
                #     "cost_estimate": None,
                #     "max_tokens": None,
                # },
                # {
                #     "run_name": Prompts.few_shot_no_sources_no_list_name,
                #     "model_type": "gpt-4",
                #     "classifier": NoSourcesNoListGenePairGPTClassifier(
                #         gene_pair_dataset,
                #         Prompts.few_shot_no_sources_no_list_name,
                #         "gpt-4",
                #         repeat_n_times=2,
                #     ),
                #     "cost_estimate": None,
                #     "max_tokens": None,
                # },
                # {
                #     "run_name": Prompts.zero_shot_no_sources_no_list_cot_name,
                #     "model_type": "gpt-4",
                #     "classifier": NoSourcesNoListGenePairGPTClassifier(
                #         gene_pair_dataset,
                #         Prompts.zero_shot_no_sources_no_list_cot_name,
                #         "gpt-4",
                #         repeat_n_times=2,
                #     ),
                #     "cost_estimate": None,
                #     "max_tokens": None,
                # },
                # {
                #     "run_name": Prompts.few_shot_no_sources_no_list_cot_name,
                #     "model_type": "gpt-4",
                #     "classifier": NoSourcesNoListGenePairGPTClassifier(
                #         gene_pair_dataset,
                #         Prompts.few_shot_no_sources_no_list_cot_name,
                #         "gpt-4",
                #         repeat_n_times=2,
                #     ),
                #     "cost_estimate": None,
                #     "max_tokens": None,
                # },
                # {
                #     "run_name": Prompts.zero_shot_no_sources_name,
                #     "model_type": "gpt-4",
                #     "classifier": NoSourcesGenePairGPTClassifier(
                #         gene_pair_dataset,
                #         Prompts.zero_shot_no_sources_name,
                #         "gpt-4",
                #         repeat_n_times=2,
                #     ),
                #     "cost_estimate": None,
                #     "max_tokens": None,
                # },
                # {
                #     "run_name": Prompts.few_shot_no_sources_name,
                #     "model_type": "gpt-4",
                #     "classifier": NoSourcesGenePairGPTClassifier(
                #         gene_pair_dataset,
                #         Prompts.few_shot_no_sources_name,
                #         "gpt-4",
                #         repeat_n_times=2,
                #     ),
                #     "cost_estimate": None,
                #     "max_tokens": None,
                # },
                # {
                #     "run_name": Prompts.zero_shot_no_sources_cot_name,
                #     "model_type": "gpt-4",
                #     "classifier": NoSourcesGenePairGPTClassifier(
                #         gene_pair_dataset,
                #         Prompts.zero_shot_no_sources_cot_name,
                #         "gpt-4",
                #         repeat_n_times=2,
                #     ),
                #     "cost_estimate": None,
                #     "max_tokens": None,
                # },
                # {
                #     "run_name": Prompts.few_shot_no_sources_cot_name,
                #     "model_type": "gpt-4",
                #     "classifier": NoSourcesGenePairGPTClassifier(
                #         gene_pair_dataset,
                #         Prompts.few_shot_no_sources_cot_name,
                #         "gpt-4",
                #         repeat_n_times=2,
                #     ),
                #     "cost_estimate": None,
                #     "max_tokens": None,
                # },
                # {
                #     "run_name": Prompts.zero_shot_treatment_source_no_list_name,
                #     "model_type": "gpt-4",
                #     "classifier": TreatmentSourceNoListGPTClassifier(
                #         treatment_source_dataset,
                #         Prompts.zero_shot_treatment_source_no_list_name,
                #         "gpt-4",
                #         repeat_n_times=2,
                #     ),
                #     "cost_estimate": None,
                #     "max_tokens": None,
                # },
                # {
                #     "run_name": Prompts.few_shot_treatment_source_no_list_name,
                #     "model_type": "gpt-4",
                #     "classifier": TreatmentSourceNoListGPTClassifier(
                #         treatment_source_dataset,
                #         Prompts.few_shot_treatment_source_no_list_name,
                #         "gpt-4",
                #         repeat_n_times=2,
                #     ),
                #     "cost_estimate": None,
                #     "max_tokens": None,
                # },
                # {
                #     "run_name": Prompts.zero_shot_treatment_source_no_list_cot_name,
                #     "model_type": "gpt-4",
                #     "classifier": TreatmentSourceNoListGPTClassifier(
                #         treatment_source_dataset,
                #         Prompts.zero_shot_treatment_source_no_list_cot_name,
                #         "gpt-4",
                #         repeat_n_times=2,
                #     ),
                #     "cost_estimate": None,
                #     "max_tokens": None,
                # },
                # {
                #     "run_name": Prompts.few_shot_treatment_source_no_list_cot_name,
                #     "model_type": "gpt-4",
                #     "classifier": TreatmentSourceNoListGPTClassifier(
                #         treatment_source_dataset,
                #         Prompts.few_shot_treatment_source_no_list_cot_name,
                #         "gpt-4",
                #         repeat_n_times=2,
                #     ),
                #     "cost_estimate": None,
                #     "max_tokens": None,
                # },
                # {
                #     "run_name": Prompts.zero_shot_treatment_source_name,
                #     "model_type": "gpt-4",
                #     "classifier": TreatmentSourceGPTClassifier(
                #         treatment_source_dataset,
                #         Prompts.zero_shot_treatment_source_name,
                #         "gpt-4",
                #         repeat_n_times=2,
                #     ),
                #     "cost_estimate": None,
                #     "max_tokens": None,
                # },
                # {
                #     "run_name": Prompts.few_shot_treatment_source_name,
                #     "model_type": "gpt-4",
                #     "classifier": TreatmentSourceGPTClassifier(
                #         treatment_source_dataset,
                #         Prompts.few_shot_treatment_source_name,
                #         "gpt-4",
                #         repeat_n_times=2,
                #     ),
                #     "cost_estimate": None,
                #     "max_tokens": None,
                # },
                # {
                #     "run_name": Prompts.zero_shot_treatment_source_cot_name,
                #     "model_type": "gpt-4",
                #     "classifier": TreatmentSourceGPTClassifier(
                #         treatment_source_dataset,
                #         Prompts.zero_shot_treatment_source_cot_name,
                #         "gpt-4",
                #         repeat_n_times=2,
                #     ),
                #     "cost_estimate": None,
                #     "max_tokens": None,
                # },
                # {
                #     "run_name": Prompts.few_shot_treatment_source_cot_name,
                #     "model_type": "gpt-4",
                #     "classifier": TreatmentSourceGPTClassifier(
                #         treatment_source_dataset,
                #         Prompts.few_shot_treatment_source_cot_name,
                #         "gpt-4",
                #         repeat_n_times=2,
                #     ),
                #     "cost_estimate": None,
                #     "max_tokens": None,
                # },
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

    def save_diagrams(self):
        """
        Save diagrams.
        """
        for run in self.run_configuration["runs"]:
            run["classifier"].base_dataset.diagrams(run["classifier"].save_dir)

    def predict(self):
        """
        Predict for all configs.
        """
        for run in self.run_configuration["runs"]:
            run["classifier"].predict()

    def results(self):
        """
        Calculate results.
        """

        def process_dummy_classifier(df, dataset, accuracy_score):
            cls = DummyClassifier().fit(dataset.df["index"], dataset.df["y_pred"])
            y_pred = cls.predict(dataset.df["y_pred"])
            y_true = dataset.df["y_true"]
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "accuracy": accuracy_score(y_true, y_pred),
                            "f1_samples": f1_score(y_true, y_pred, average="samples"),
                            "f1_macro": f1_score(y_true, y_pred, average="macro"),
                            "f1_micro": f1_score(y_true, y_pred, average="micro"),
                            "precision_samples": precision_score(
                                y_true, y_pred, average="samples"
                            ),
                            "precision_macro": precision_score(
                                y_true, y_pred, average="macro"
                            ),
                            "precision_micro": precision_score(
                                y_true, y_pred, average="micro"
                            ),
                            "recall_samples": recall_score(
                                y_true, y_pred, average="samples"
                            ),
                            "recall_macro": recall_score(
                                y_true, y_pred, average="macro"
                            ),
                            "recall_micro": recall_score(
                                y_true, y_pred, average="micro"
                            ),
                        }
                    ),
                ]
            )

        def process_results(df):
            df = pd.concat([df, run["classifier"].base_dataset.aggregate_results()])
            df["model_name"] = run["model_type"]
            df["run_type"] = (
                "zero_shot" if "zero_shot" in run["run_name"] else "few_shot"
            )
            df["run_cot_type"] = "cot" if "cot" in run["run_name"] else "not_cot"

        for run in self.run_configuration["runs"]:
            if isinstance(run["classifier"].base_dataset, GenePairDataset):
                process_results(self._gene_pair_results)
            else:
                process_results(self._treatment_source_results)

        process_dummy_classifier(
            self._treatment_source_results,
            self._treatment_source_dataset,
            metrics.accuracy_score,
        )
        process_dummy_classifier(
            self._gene_pair_results, self._gene_pair_dataset, util.accuracy_score
        )

    def run_all(self):
        """
        Run all components.
        """
        self.calculate_costs()
        self.save_example_prompts()
        self.predict()
        self.save_diagrams()
        self.results()

    @property
    def treatment_source_results(self) -> pd.DataFrame:
        """
        Get results
        """
        return self._treatment_source_results

    @property
    def gene_pair_results(self) -> pd.DataFrame:
        """
        Get results
        """
        return self._gene_pair_results
