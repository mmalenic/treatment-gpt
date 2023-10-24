from collections import Counter
import random
from collections import Counter
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import (
    hamming_loss,
    classification_report,
)
from sklearn.preprocessing import MultiLabelBinarizer

from classifier.util import accuracy_score
from dataset.load_protect import LoadProtect
from dataset.utils import process_plus
from dataset.utils import (
    save_fig,
    heatmap_for_cls_report,
    results,
)


class GenePairDataset:
    """
    Loads all protect data.
    """

    def __init__(
        self,
        from_protect: LoadProtect,
        remove_empty_sources: bool = False,
        split_to_n_treatments: Optional[int] = 3,
        remove_generic_cancer_types=None,
        remove_generic_treatments=None,
        **kwargs,
    ) -> None:
        """
        Initialize this class.

        :param from_protect: load protect data from this class.
        """
        if remove_generic_cancer_types is None:
            remove_generic_cancer_types = ["cancer"]
        if remove_generic_treatments is None:
            remove_generic_treatments = [
                "chemotherapy",
                "chemotherapy + everolimus + trastuzumab",
            ]

        self.__dict__.update(kwargs)

        self._dataset = []
        self._df = pd.DataFrame()
        self._from_protect = from_protect
        self._remove_empty_sources = remove_empty_sources
        self._remove_generic_cancer_types = remove_generic_cancer_types
        self._remove_generic_treatments = remove_generic_treatments
        self._split_to_n_treatments = split_to_n_treatments
        self._all_treatments = {}
        self._original_labels = {}

        self._binarizer = MultiLabelBinarizer()

    def load(self):
        """
        Load the dataset.
        """
        for index, row in self._from_protect.df().iterrows():
            if self._remove_generic_cancer_types:
                if row["cancer_type"] in self._remove_generic_cancer_types:
                    continue

            treatments = (
                row["treatment_with_text_sources_x"]
                + row["treatment_with_text_sources_y"]
            )
            treatments = [
                {"treatment": x[0], "source": x[1], "level": x[2]}
                for i, x in enumerate(treatments)
                if not any((x[0] == y[0] for y in treatments[:i]))
            ]

            if self._remove_empty_sources:
                treatments = [
                    x
                    for x in treatments
                    if x["source"] is not None and x["source"] != ""
                ]

            random.shuffle(treatments)
            if self._split_to_n_treatments is not None:
                split_to = np.ceil(len(treatments) / self._split_to_n_treatments)
                treatments = [
                    x.tolist() for x in np.array_split(np.array(treatments), split_to)
                ]
            else:
                treatments = [treatments]

            for treatment_sublist in treatments:
                y_true = [x["treatment"] for x in treatment_sublist]

                if self._remove_generic_treatments:
                    y_true = [
                        x
                        for x in y_true
                        if x.lower() not in self._remove_generic_treatments
                    ]

                if len(y_true) == 0 or y_true is None:
                    continue

                all_treatments = self._from_protect.treatments_and_sources()
                all_treatments = list([x for x in all_treatments if x[0] not in y_true])

                if self._remove_empty_sources:
                    all_treatments = [
                        x for x in all_treatments if x[1] is not None and x[1] != ""
                    ]

                random.shuffle(all_treatments)
                treatment_sublist += [
                    {"treatment": x[0], "source": x[1], "level": x[2]}
                    for x in all_treatments[: len(y_true)]
                ]

                if self._remove_generic_treatments:
                    treatment_sublist = [
                        x
                        for x in treatment_sublist
                        if x["treatment"].lower() not in self._remove_generic_treatments
                    ]

                random.shuffle(treatment_sublist)

                self._all_treatments.update(
                    {x["treatment"]: None for x in treatment_sublist}
                )

                y_true_processed = []
                for y in y_true:
                    process = process_plus(y, self)
                    self._original_labels[process] = y
                    y_true_processed.append(process)

                if treatment_sublist and y_true:
                    self._dataset.append(
                        {
                            "index": index,
                            "cancer_type": row["cancer_type"],
                            "gene_x": row["gene_x"],
                            "gene_y": row["gene_y"],
                            "p_val": row["p_val"],
                            "correlation_type": row["correlation_type"],
                            "odds": row["odds"],
                            "treatments": treatment_sublist,
                            "y_true": y_true_processed,
                            "y_true_original": y_true,
                            "y_pred": np.nan,
                        }
                    )

        self._df = pd.DataFrame(self._dataset)
        self._binarizer.fit([[process_plus(x, self) for x in self.all_treatments]])

    def treatments_for(self, cancer_type) -> List[str]:
        """
        Get the treatments that are present in the true labels of the dataset for a given cancer type.
        """
        treatments = {}

        for data in self._dataset:
            if data["cancer_type"] == cancer_type:
                treatments.update({treatment: None for treatment in data["y_true"]})

        return list(treatments.keys())

    def label_counts(self) -> pd.DataFrame:
        """
        Get the label counts.
        """
        labels = [
            z.strip()
            for x in self._df["y_true_original"].to_list()
            for y in x
            for z in y.split("+")
        ]
        counter = Counter(labels)

        return (
            pd.DataFrame.from_dict(counter, orient="index")
            .reset_index()
            .sort_values(by=0, ascending=False)
        )

    def gene_counts(self) -> pd.DataFrame:
        """
        Get the gene counts.
        """
        genes = self._df["gene_x"].to_list() + self._df["gene_y"].to_list()
        counter = Counter(genes)

        return (
            pd.DataFrame.from_dict(counter, orient="index")
            .reset_index()
            .sort_values(by=0, ascending=False)
        )

    def results(self, x) -> pd.DataFrame:
        """
        Compute the results
        """

        def treatment_for_level(y, level):
            treatments = [
                treatment["level"]
                for treatment in x["treatments"]
                if treatment["treatment"].lower() == y
            ]
            if len(treatments) == 0:
                return False
            else:
                return treatments[0] == level

        def score(y_true, y_pred, suffix):
            try:
                y_true = self._binarizer.transform([y_true])
                y_pred = self._binarizer.transform([y_pred])

                x["hamming_loss" + suffix] = hamming_loss(y_true, y_pred)
                x["accuracy_score" + suffix] = accuracy_score(y_true, y_pred)
            except (TypeError, ValueError) as e:
                print("error:", e)
                x["hamming_loss" + suffix] = 0
                x["accuracy_score" + suffix] = 0

        x["y_true_a_level"] = [y for y in x["y_true"] if treatment_for_level(y, "A")]
        x["y_true_b_level"] = [y for y in x["y_true"] if treatment_for_level(y, "B")]
        x["y_pred_a_level"] = [y for y in x["y_pred"] if treatment_for_level(y, "A")]
        x["y_pred_b_level"] = [y for y in x["y_pred"] if treatment_for_level(y, "B")]

        score(x["y_true"], x["y_pred"], "")

        if len(x["y_true_a_level"]) != 0:
            score(x["y_true_a_level"], x["y_pred_a_level"], "_a_level")

        if len(x["y_true_b_level"]) != 0:
            score(x["y_true_b_level"], x["y_pred_b_level"], "_b_level")

        return x

    def aggregate_results(self):
        """
        Get aggregate results.
        """
        y_true = self._binarizer.transform(self._df["y_true"].tolist())
        y_pred = self._binarizer.transform(
            self.process_predictions(self._df["y_pred"].tolist())
        )

        return results(y_true, y_pred, accuracy_score)

    @staticmethod
    def process_predictions(y_pred, should_be_list=True):
        """
        Process and wrap predictions.
        """
        out = []
        for x in y_pred:
            if isinstance(x, list) and should_be_list:
                out.append([str(y) for y in x])
            else:
                out.append(str(x))

        return out

    def diagrams(self, save_to: str):
        """
        Save all diagrams.
        """

        def cls_report(x):
            return pd.DataFrame(
                classification_report(
                    self._binarizer.transform(x["y_true"].tolist()),
                    self._binarizer.transform(
                        self.process_predictions(x["y_pred"].tolist())
                    ),
                    output_dict=True,
                    target_names=[
                        self._original_labels[y] for y in self._binarizer.classes_
                    ],
                )
            ).transpose()

        def add_cancer_types_code(x, from_protect: LoadProtect):
            """
            Add cancer type code to dataframe.
            """
            x["cancer_type_code"] = from_protect.cancer_types.cancer_type_code(
                x["cancer_type"]
            )
            return x

        def heat_map(x, colour, save_to, height, width, transpose=True):
            x = x.set_index("Treatment")

            return heatmap_for_cls_report(
                x,
                colour,
                save_to,
                x_label="Treatment",
                y_label="Score",
                height=height,
                width=width,
                transpose=transpose,
            )

        def plot_heatmaps(df, save_to):
            """
            Plot all heatmaps.
            """
            df.groupby(["cancer_type"]).apply(
                lambda x: heat_map(
                    x,
                    "Blues",
                    f"{save_to}/heatmaps/{x['cancer_type'].iloc[0]}_small_transposed.png",
                    12,
                    5,
                )
            )
            df.groupby(["cancer_type"]).apply(
                lambda x: heat_map(
                    x,
                    "Blues",
                    f"{save_to}/heatmaps/{x['cancer_type'].iloc[0]}_medium_transposed.png",
                    12,
                    10,
                )
            )
            df.groupby(["cancer_type"]).apply(
                lambda x: heat_map(
                    x,
                    "Blues",
                    f"{save_to}/heatmaps/{x['cancer_type'].iloc[0]}_large_transposed.png",
                    12,
                    15,
                )
            )
            df.groupby(["cancer_type"]).apply(
                lambda x: heat_map(
                    x,
                    "Blues",
                    f"{save_to}/heatmaps/{x['cancer_type'].iloc[0]}_small.png",
                    5,
                    12,
                    transpose=False,
                )
            )
            df.groupby(["cancer_type"]).apply(
                lambda x: heat_map(
                    x,
                    "Blues",
                    f"{save_to}/heatmaps/{x['cancer_type'].iloc[0]}_medium.png",
                    10,
                    12,
                    transpose=False,
                )
            )
            df.groupby(["cancer_type"]).apply(
                lambda x: heat_map(
                    x,
                    "Blues",
                    f"{save_to}/heatmaps/{x['cancer_type'].iloc[0]}_large.png",
                    15,
                    12,
                    transpose=False,
                )
            )

        Path(save_to).mkdir(exist_ok=True, parents=True)
        heatmaps_save = Path(f"{save_to}/heatmaps/")
        heatmaps_save.mkdir(exist_ok=True, parents=True)

        df = self.df.copy()
        df.loc[df["correlation_type"] == "none", "correlation_type"] = "no correlation"

        df_cancer_types = df.apply(
            lambda x: add_cancer_types_code(x, self._from_protect), axis=1
        )
        has_cor = df.loc[df["correlation_type"] != "no correlation"]
        mutually_exclusive = df.loc[df["correlation_type"] == "mutual exclusivity"]
        cooccuring = df.loc[df["correlation_type"] == "co-occurrence"]

        melt_correlation_type = df.melt(
            id_vars=["correlation_type"],
            value_vars=["accuracy_score_a_level", "accuracy_score_b_level"],
        )
        melt_correlation_type = melt_correlation_type.rename(
            columns={"variable": "Evidence level"}
        )
        melt_correlation_type.loc[
            melt_correlation_type["Evidence level"] == "accuracy_score_a_level",
            "Evidence level",
        ] = "A"
        melt_correlation_type.loc[
            melt_correlation_type["Evidence level"] == "accuracy_score_b_level",
            "Evidence level",
        ] = "B"

        melt_level = df.melt(
            value_vars=["accuracy_score_a_level", "accuracy_score_b_level"],
        )
        melt_level = melt_level.rename(columns={"variable": "Evidence level"})
        melt_level.loc[
            melt_level["Evidence level"] == "accuracy_score_a_level",
            "Evidence level",
        ] = "A"
        melt_level.loc[
            melt_level["Evidence level"] == "accuracy_score_b_level",
            "Evidence level",
        ] = "B"

        group_by_cancer_type = (
            df.groupby(["cancer_type"]).apply(cls_report).reset_index()
        )
        group_by_cancer_type = group_by_cancer_type.rename(
            columns={"level_1": "Treatment"}
        )
        group_by_cancer_type = group_by_cancer_type[
            group_by_cancer_type.apply(
                lambda x: True
                if process_plus(x["Treatment"], self)
                in self.treatments_for(x["cancer_type"])
                else False,
                axis=1,
            )
        ]

        plot_heatmaps(group_by_cancer_type, save_to)

        plt.clf()
        plt.figure()
        plot = sns.barplot(df, x="correlation_type", y="accuracy_score")
        plot.set(xlabel="Correlation type", ylabel="Accuracy score")
        save_fig(f"{save_to}/accuracy_score.png")

        plt.clf()
        plt.figure()
        plot = sns.lmplot(
            data=has_cor, y="accuracy_score", x="p_val", hue="correlation_type"
        )
        plt.ylabel("Accuracy score")
        plt.xlabel("p-value")
        plt.subplots_adjust(top=0.9)
        plot._legend.set_title("Correlation type")
        save_fig(f"{save_to}/accuracy_p_val_for_correlation.png", tight=False)

        plt.clf()
        plt.figure()
        plot = sns.lmplot(
            data=has_cor, y="accuracy_score", x="odds", hue="correlation_type"
        )
        plt.ylabel("Accuracy score")
        plt.xlabel("Odds")
        plt.subplots_adjust(top=0.9)
        plot._legend.set_title("Correlation type")
        save_fig(f"{save_to}/accuracy_odds_for_correlation.png", tight=False)

        plt.clf()
        plt.figure()
        plot = sns.lmplot(
            data=mutually_exclusive,
            y="accuracy_score",
            x="p_val",
            hue="correlation_type",
        )
        plt.ylabel("Accuracy score")
        plt.xlabel("p-value")
        plt.subplots_adjust(top=0.9)
        plot._legend.set_title("Correlation type")
        save_fig(f"{save_to}/accuracy_p_value_mutually_exclusive.png", tight=False)

        plt.clf()
        plt.figure()
        plot = sns.lmplot(
            data=mutually_exclusive,
            y="accuracy_score",
            x="odds",
            hue="correlation_type",
        )
        plt.ylabel("Accuracy score")
        plt.xlabel("Odds")
        plt.subplots_adjust(top=0.9)
        plot._legend.set_title("Correlation type")
        save_fig(f"{save_to}/accuracy_odds_mutually_exclusive.png", tight=False)

        plt.clf()
        plt.figure()
        plot = sns.lmplot(
            data=cooccuring, y="accuracy_score", x="p_val", hue="correlation_type"
        )
        plt.ylabel("Accuracy score")
        plt.xlabel("p-value")
        plt.subplots_adjust(top=0.9)
        plot._legend.set_title("Correlation type")
        save_fig(f"{save_to}/accuracy_p_value_co_occurring.png", tight=False)

        plt.clf()
        plt.figure()
        plot = sns.lmplot(
            data=cooccuring, y="accuracy_score", x="odds", hue="correlation_type"
        )
        plt.ylabel("Accuracy score")
        plt.xlabel("Odds")
        plt.subplots_adjust(top=0.9)
        plot._legend.set_title("Correlation type")
        save_fig(f"{save_to}/accuracy_odds_co_occurring.png", tight=False)

        plt.clf()
        plt.figure()
        plot = sns.barplot(
            melt_correlation_type, x="correlation_type", y="value", hue="Evidence level"
        )
        plot.set(ylabel="Accuracy score", xlabel="Correlation type")
        save_fig(f"{save_to}/accuracy_level_correlation_type.png")

        plt.clf()
        plt.figure()
        plot = sns.barplot(df_cancer_types, x="cancer_type_code", y="accuracy_score")
        plot.set(ylabel="Accuracy score", xlabel="Cancer type")
        plt.subplots_adjust(bottom=0.3)
        plt.xticks(rotation=90)
        plt.subplots_adjust(top=0.9)
        plt.title("a", x=0, fontweight="bold", fontsize=16)
        save_fig(f"{save_to}/accuracy_cancer_type.png", tight=False)

        plt.clf()

    @property
    def all_treatments(self) -> List[str]:
        """
        Return the dataframe of the dataset.
        """
        return list([x.lower() for x in self._all_treatments.keys()])

    def get_all_treatments(self) -> List[str]:
        """
        Return the dataframe of the dataset.
        """
        return list([x for x in self._all_treatments.keys()])

    @property
    def df(self) -> pd.DataFrame:
        """
        Return the dataframe of the dataset.
        """
        return self._df

    @df.setter
    def df(self, df):
        """
        Set the df.
        """
        self._df = df

    def alternative_name(self, name: str) -> str:
        """
        Find the canonical name.
        """
        out_name = self._from_protect.alternative_names.find_name(name)

        if out_name is None:
            return name
        return out_name

    def add_prediction(self, prediction, pos):
        self._df.iloc[pos, self._df.columns.get_loc("y_pred")] = prediction

    def dataset(self):
        """
        Return the dataset.
        """
        return self._dataset
