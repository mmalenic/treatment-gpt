import os.path
from collections import Counter
from pathlib import Path
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
from dataset.load_protect import LoadProtect
from dataset.treatment_source_dataset import TreatmentSourceDataset
from dataset.gene_pair_dataset import GenePairDataset

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from itertools import chain

from dataset.utils import results, save_fig


class RunConfiguration:
    def __init__(
        self,
        load: LoadProtect,
    ):
        """
        Initialize this class.
        """

        def load_dataset(dataset):
            dataset.load()
            return dataset

        self._load = load

        self._with_gene_pair_dataset = lambda: load_dataset(
            GenePairDataset(load, split_to_n_treatments=None)
        )
        self._with_treatment_source_dataset = lambda: load_dataset(
            TreatmentSourceDataset(load)
        )

        self._treatment_source_results = pd.DataFrame()
        self._gene_pair_results = pd.DataFrame()

        self._run_configuration = {
            "runs": [
                {
                    "run_name": Prompts.zero_shot_no_sources_no_list_name,
                    "model_type": "gpt-3.5-turbo",
                    "classifier": NoSourcesNoListGenePairGPTClassifier(
                        self._with_gene_pair_dataset(),
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
                        self._with_gene_pair_dataset(),
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
                        self._with_gene_pair_dataset(),
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
                        self._with_gene_pair_dataset(),
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
                        self._with_gene_pair_dataset(),
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
                        self._with_gene_pair_dataset(),
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
                        self._with_gene_pair_dataset(),
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
                        self._with_gene_pair_dataset(),
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
                        self._with_treatment_source_dataset(),
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
                        self._with_treatment_source_dataset(),
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
                        self._with_treatment_source_dataset(),
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
                        self._with_treatment_source_dataset(),
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
                        self._with_treatment_source_dataset(),
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
                        self._with_treatment_source_dataset(),
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
                        self._with_treatment_source_dataset(),
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
                        self._with_treatment_source_dataset(),
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
                        self._with_gene_pair_dataset(),
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
                        self._with_gene_pair_dataset(),
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
                        self._with_gene_pair_dataset(),
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
                        self._with_gene_pair_dataset(),
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
                        self._with_gene_pair_dataset(),
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
                        self._with_gene_pair_dataset(),
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
                        self._with_gene_pair_dataset(),
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
                        self._with_gene_pair_dataset(),
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
                        self._with_treatment_source_dataset(),
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
                        self._with_treatment_source_dataset(),
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
                        self._with_treatment_source_dataset(),
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
                        self._with_treatment_source_dataset(),
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
                        self._with_treatment_source_dataset(),
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
                        self._with_treatment_source_dataset(),
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
                        self._with_treatment_source_dataset(),
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
                        self._with_treatment_source_dataset(),
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

    def save_diagrams(self):
        """
        Save diagrams.
        """

        def aggregate_dataset(instance_type, model_type):
            return pd.concat(
                [
                    x["classifier"].base_dataset.df
                    for x in self.run_configuration["runs"]
                    if x["model_type"] == model_type
                    and isinstance(x["classifier"].base_dataset, instance_type)
                ]
            )

        def with_dataset(create_dataset, new_data, save_dir):
            dataset = create_dataset()
            dataset.df = new_data
            dataset.diagrams("data/results/" + save_dir)
            return dataset

        for run in self.run_configuration["runs"]:
            run["classifier"].base_dataset.diagrams(
                os.path.join(run["classifier"].save_dir, "diagrams")
            )

        gene_pair_datasets_3_5 = with_dataset(
            self._with_gene_pair_dataset,
            aggregate_dataset(GenePairDataset, "gpt-3.5-turbo"),
            "gene_pair_gpt_3_5",
        )
        gene_pair_datasets_4 = with_dataset(
            self._with_gene_pair_dataset,
            aggregate_dataset(GenePairDataset, "gpt-4"),
            "gene_pair_gpt_4",
        )

        treatment_source_datasets_3_5 = with_dataset(
            self._with_treatment_source_dataset,
            aggregate_dataset(TreatmentSourceDataset, "gpt-3.5-turbo"),
            "treatment_gpt_3_5",
        )
        treatment_source_datasets_4 = with_dataset(
            self._with_treatment_source_dataset,
            aggregate_dataset(TreatmentSourceDataset, "gpt-4"),
            "treatment_gpt_4",
        )

        gene_pair_datasets_all = with_dataset(
            self._with_gene_pair_dataset,
            pd.concat([gene_pair_datasets_3_5.df, gene_pair_datasets_4.df]),
            "gene_pair_all",
        )
        treatment_source_datasets_all = with_dataset(
            self._with_treatment_source_dataset,
            pd.concat(
                [treatment_source_datasets_3_5.df, treatment_source_datasets_4.df]
            ),
            "treatment_all",
        )

        self.run_configuration["all"] = [
            gene_pair_datasets_3_5,
            gene_pair_datasets_4,
            gene_pair_datasets_all,
            treatment_source_datasets_3_5,
            treatment_source_datasets_4,
            treatment_source_datasets_all,
        ]

    async def predict(self):
        """
        Predict for all configs.
        """
        for run in self.run_configuration["runs"]:
            await run["classifier"].predict()

    def results(self, from_path: bool = False):
        """
        Calculate results.
        """

        if from_path and Path("data/gene_pair_results.xlsx").exists():
            self._gene_pair_results = pd.read_excel("data/gene_pair_results.xlsx")
            self.results_diagram(self._gene_pair_results, "data/gene_pair_results_")

        if from_path and Path("data/treatment_source_results.xlsx").exists():
            self._treatment_source_results = pd.read_excel(
                "data/treatment_source_results.xlsx"
            )
            self.results_diagram(
                self._treatment_source_results, "data/treatment_source_results_"
            )

        if from_path:
            return

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

            try:
                y_true = dataset._binarizer.transform(y_true)
                y_pred = dataset._binarizer.transform(y_pred)
            except AttributeError:
                pass

            out = results(y_true, y_pred, accuracy_score, sample_wise)
            out["Model name"] = "dummy"

            df = pd.concat(
                [df, out],
                ignore_index=True,
            )

            return df

        def label_report(x):
            y = [x["Run type"]]
            if x["List of answers"] == "yes":
                y.append("list hint")
            if x["COT"] == "COT":
                y.append("COT")

            return " + ".join(y)

        def process_results(df, run):
            out = run["classifier"].base_dataset.aggregate_results()
            out["Model name"] = run["model_type"]
            out["Run type"] = (
                "zero shot" if "zero_shot" in run["run_name"] else "few shot"
            )
            out["List of answers"] = "no" if "no_list" in run["run_name"] else "yes"
            out["COT"] = "COT" if "cot" in run["run_name"] else "no COT"
            out["Report name"] = out.apply(label_report, axis=1)

            df = pd.concat(
                [df, out],
                ignore_index=True,
            )

            return df

        treatment_source_dataset = pd.DataFrame()
        gene_pair_dataset = pd.DataFrame()

        for run in self.run_configuration["runs"]:
            dataset = run["classifier"].base_dataset
            if isinstance(dataset, GenePairDataset):
                self._gene_pair_results = process_results(self._gene_pair_results, run)
                gene_pair_dataset = dataset
            else:
                self._treatment_source_results = process_results(
                    self._treatment_source_results, run
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

        if not Path("data/gene_pair_results.xlsx").exists():
            self._gene_pair_results.to_excel("data/gene_pair_results.xlsx")
            self.results_diagram(self._gene_pair_results, "data/gene_pair_results.png")

        if not Path("data/treatment_source_results.xlsx").exists():
            self._treatment_source_results.to_excel(
                "data/treatment_source_results.xlsx"
            )
            self.results_diagram(
                self._treatment_source_results, "data/treatment_source_results"
            )

    @staticmethod
    def results_diagram(results, save_to):
        results = results[results["Model name"] != "dummy"]
        g = sns.catplot(
            data=results,
            col="Model name",
            y="accuracy",
            x="COT",
            hue="Run type",
            capsize=0.1,
            palette="YlGnBu_d",
            errorbar="se",
            kind="point",
            height=6,
            aspect=0.75,
        )
        g.despine(left=True)
        g.fig.subplots_adjust(top=0.9)
        save_fig(save_to + "cot_comparison.png")

        g = sns.catplot(
            data=results,
            x="Model name",
            y="accuracy",
            hue="COT",
            col="Run type",
            capsize=0.1,
            palette="YlGnBu_d",
            errorbar="se",
            kind="point",
            height=6,
            aspect=0.75,
        )
        g.despine(left=True)
        g.fig.subplots_adjust(top=0.9)
        save_fig(save_to + "model_comparison.png", tight=False)

        g = sns.catplot(
            data=results,
            x="List of answers",
            y="accuracy",
            hue="COT",
            col="Run type",
            capsize=0.1,
            palette="YlGnBu_d",
            errorbar="se",
            kind="point",
            height=6,
            aspect=0.75,
        )
        g.despine(left=True)
        g.fig.subplots_adjust(top=0.9)
        save_fig(save_to + "list_of_answers.png", tight=False)

    async def run_all(self, from_path: bool = False, save_diagrams: bool = True):
        """
        Run all components.
        """
        self.calculate_costs()
        self.save_example_prompts()

        if not from_path:
            await self.predict()

        self.results(from_path)

        if save_diagrams:
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
