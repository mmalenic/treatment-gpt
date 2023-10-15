import os.path
from collections import Counter
from typing import Callable

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
from itertools import chain

from dataset.utils import results


class RunConfiguration:
    def __init__(
        self,
        with_gene_pair_dataset: Callable[[], GenePairDataset],
        with_treatment_source_dataset: Callable[[], TreatmentSourceDataset],
    ):
        """
        Initialize this class.
        """

        self._with_gene_pair_dataset = with_gene_pair_dataset
        self._with_treatment_source_dataset = with_treatment_source_dataset

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
                        with_gene_pair_dataset(),
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
                        with_treatment_source_dataset(),
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
            run["classifier"].base_dataset.diagrams(
                os.path.join(run["classifier"].save_dir, "diagrams")
            )

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

        def process_dummy_classifier(df, dataset, accuracy_score, sample_wise):
            if dataset.df["y_pred"].isnull().all():
                return df

            y_true_flat = dataset.df["y_true"].tolist()
            if any(isinstance(x, list) for x in y_true_flat):
                y_true_flat = list(chain(*y_true_flat))

            counts = Counter(y_true_flat)
            most_common = counts.most_common(1)[0][0]

            y_true = dataset.df["y_true"].tolist()
            y_pred = [
                [most_common] * len(x) if isinstance(x, list) else most_common
                for x in y_true
            ]

            if y_pred:
                df = pd.concat(
                    [df, results(y_true, y_pred, accuracy_score, sample_wise)],
                    ignore_index=True,
                )

            df["model_name"] = run["dummy"]

            return df

        def process_results(df):
            df = pd.concat(
                [df, run["classifier"].base_dataset.aggregate_results()],
                ignore_index=True,
            )
            df["model_name"] = run["model_type"]
            df["run_type"] = (
                "zero_shot" if "zero_shot" in run["run_name"] else "few_shot"
            )
            df["run_cot_type"] = "cot" if "cot" in run["run_name"] else "not_cot"

            return df

        treatment_source_dataset = pd.DataFrame()
        gene_pair_dataset = pd.DataFrame()
        for run in self.run_configuration["runs"]:
            dataset = run["classifier"].base_dataset
            if isinstance(run["classifier"].base_dataset, GenePairDataset):
                self._gene_pair_results = process_results(self._gene_pair_results)
                gene_pair_dataset = dataset
            else:
                self._treatment_source_results = process_results(
                    self._treatment_source_results
                )
                treatment_source_dataset = dataset

        self._treatment_source_results = process_dummy_classifier(
            self._treatment_source_results,
            treatment_source_dataset,
            metrics.accuracy_score,
            False,
        )
        self._gene_pair_results = process_dummy_classifier(
            self._gene_pair_results,
            gene_pair_dataset,
            util.accuracy_score,
            True,
        )

    def run_all(self):
        """
        Run all components.
        """
        self.calculate_costs()
        self.save_example_prompts()
        self.predict()
        self.results()
        self.save_diagrams()

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
