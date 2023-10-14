import itertools
from pathlib import Path
from typing import Optional, List
import seaborn as sns

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import hamming_loss, classification_report
from sklearn.preprocessing import MultiLabelBinarizer

from classifier.util import accuracy_score
from dataset.load_protect import LoadProtect
import random

from dataset.diagram_utils import (
    save_fig,
    heatmap_for_cls_report,
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
        **kwargs,
    ) -> None:
        """
        Initialize this class.

        :param from_protect: load protect data from this class.
        """
        self.__dict__.update(kwargs)

        self._dataset = []
        self._df = pd.DataFrame()
        self._from_protect = from_protect
        self._remove_empty_sources = remove_empty_sources
        self._split_to_n_treatments = split_to_n_treatments
        self._all_treatments = {}

        self._binarizer = MultiLabelBinarizer()

    def load(self):
        """
        Load the dataset.
        """

        for index, row in self._from_protect.df().iterrows():
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

                random.shuffle(treatment_sublist)

                self._all_treatments.update(
                    {x["treatment"]: None for x in treatment_sublist}
                )

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
                            "y_true": [y.lower() for y in y_true],
                            "y_pred": np.nan,
                        }
                    )

        self._df = pd.DataFrame(self._dataset)
        self._binarizer.fit([[x.lower() for x in self.all_treatments]])

    def treatments_for(self, cancer_type) -> List[str]:
        """
        Get the treatments that are present in the true labels of the dataset for a given cancer type.
        """
        treatments = {}

        for data in self._dataset:
            if data["cancer_type"] == cancer_type:
                treatments.update({treatment: None for treatment in data["y_true"]})

        return list(treatments.keys())

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
            y_true = self._binarizer.transform([y_true])
            y_pred = self._binarizer.transform([y_pred])

            x["hamming_loss" + suffix] = hamming_loss(y_true, y_pred)
            x["accuracy_score" + suffix] = accuracy_score(y_true, y_pred)

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

    def diagrams(self, save_to: str):
        """
        Save all diagrams.
        """

        def cls_report(x):
            return pd.DataFrame(
                classification_report(
                    self._binarizer.transform(x["y_true"].tolist()),
                    self._binarizer.transform(x["y_pred"].tolist()),
                    output_dict=True,
                    target_names=self._binarizer.classes_,
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

        def heat_map(x, colour, save_to):
            x["cancer_type"] = f"Scores for {x['cancer_type'].iloc[0]}"
            x = x.set_index("Treatment")

            return heatmap_for_cls_report(
                x,
                colour,
                save_to,
                x_label="Treatment",
                y_label="Score",
                title=f"Scores for {x['cancer_type'].iloc[0]}",
            )

        def plot_heatmaps(df, save_to):
            """
            Plot all heatmaps.
            """
            df.groupby(["cancer_type"]).apply(
                lambda x: heat_map(
                    x,
                    "Blues",
                    f"{save_to}/heatmaps/{x['cancer_type'].iloc[0]}_blue.svg",
                )
            )
            df.groupby(["cancer_type"]).apply(
                lambda x: heat_map(
                    x, "Reds", f"{save_to}/heatmaps/{x['cancer_type'].iloc[0]}_red.svg"
                )
            )
            df.groupby(["cancer_type"]).apply(
                lambda x: heat_map(
                    x,
                    "Greens",
                    f"{save_to}/heatmaps/{x['cancer_type'].iloc[0]}_green.svg",
                )
            )
            df.groupby(["cancer_type"]).apply(
                lambda x: heat_map(
                    x,
                    "Oranges",
                    f"{save_to}/heatmaps/{x['cancer_type'].iloc[0]}_orange.svg",
                )
            )
            df.groupby(["cancer_type"]).apply(
                lambda x: heat_map(
                    x,
                    "Purples",
                    f"{save_to}/heatmaps/{x['cancer_type'].iloc[0]}_purple.svg",
                )
            )

        Path(save_to).mkdir(exist_ok=True, parents=True)
        Path(f"{save_to}/heatmaps/").mkdir(exist_ok=True, parents=True)

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

        group_by_cancer_type = (
            df.groupby(["cancer_type"]).apply(cls_report).reset_index()
        )
        group_by_cancer_type = group_by_cancer_type.rename(
            columns={"level_1": "Treatment"}
        )
        group_by_cancer_type = group_by_cancer_type[
            group_by_cancer_type.apply(
                lambda x: True
                if x["Treatment"] in self.treatments_for(x["cancer_type"])
                else False,
                axis=1,
            )
        ]

        plot_heatmaps(group_by_cancer_type, save_to)

        plt.clf()
        plt.figure()
        plot = sns.barplot(df, x="correlation_type", y="accuracy_score")
        plot.set_title("Accuracy for correlation type")
        plot.set(xlabel="Correlation type", ylabel="Accuracy score")
        save_fig(f"{save_to}/accuracy_score.svg")

        plt.clf()
        plt.figure()
        plot = sns.lmplot(
            data=has_cor, y="accuracy_score", x="p_val", hue="correlation_type"
        )
        plt.title("Accuracy and p-value for correlations")
        plt.ylabel("Accuracy score")
        plt.xlabel("p-value")
        plt.subplots_adjust(top=0.9)
        plot._legend.set_title("Correlation type")
        save_fig(f"{save_to}/accuracy_p_val_for_correlation.svg", tight=False)

        plt.clf()
        plt.figure()
        plot = sns.lmplot(
            data=has_cor, y="accuracy_score", x="odds", hue="correlation_type"
        )
        plt.title("Accuracy and odds for correlations")
        plt.ylabel("Accuracy score")
        plt.xlabel("Odds")
        plt.subplots_adjust(top=0.9)
        plot._legend.set_title("Correlation type")
        save_fig(f"{save_to}/accuracy_odds_for_correlation.svg", tight=False)

        plt.clf()
        plt.figure()
        plot = sns.lmplot(
            data=mutually_exclusive,
            y="accuracy_score",
            x="p_val",
            hue="correlation_type",
        )
        plt.title("Accuracy and p-value for mutually exclusive correlations")
        plt.ylabel("Accuracy score")
        plt.xlabel("p-value")
        plt.subplots_adjust(top=0.9)
        plot._legend.set_title("Correlation type")
        save_fig(f"{save_to}/accuracy_p_value_mutually_exclusive.svg", tight=False)

        plt.clf()
        plt.figure()
        plot = sns.lmplot(
            data=mutually_exclusive,
            y="accuracy_score",
            x="odds",
            hue="correlation_type",
        )
        plt.title("Accuracy and odds for mutually exclusive correlations")
        plt.ylabel("Accuracy score")
        plt.xlabel("Odds")
        plt.subplots_adjust(top=0.9)
        plot._legend.set_title("Correlation type")
        save_fig(f"{save_to}/accuracy_odds_mutually_exclusive.svg", tight=False)

        plt.clf()
        plt.figure()
        plot = sns.lmplot(
            data=cooccuring, y="accuracy_score", x="p_val", hue="correlation_type"
        )
        plt.title("Accuracy and p-value for co-occurring correlations")
        plt.ylabel("Accuracy score")
        plt.xlabel("p-value")
        plt.subplots_adjust(top=0.9)
        plot._legend.set_title("Correlation type")
        save_fig(f"{save_to}/accuracy_p_value_co_occurring.svg", tight=False)

        plt.clf()
        plt.figure()
        plot = sns.lmplot(
            data=cooccuring, y="accuracy_score", x="odds", hue="correlation_type"
        )
        plt.title("Accuracy and odds for co-occurring correlations")
        plt.ylabel("Accuracy score")
        plt.xlabel("Odds")
        plt.subplots_adjust(top=0.9)
        plot._legend.set_title("Correlation type")
        save_fig(f"{save_to}/accuracy_odds_co_occurring.svg", tight=False)

        plt.clf()
        plt.figure()
        plot = sns.barplot(
            melt_correlation_type, x="correlation_type", y="value", hue="Evidence level"
        )
        plot.set_title("Accuracy for evidence levels and correlation type")
        plot.set(ylabel="Accuracy score", xlabel="Correlation type")
        save_fig(f"{save_to}/accuracy_level_correlation_type.svg")

        plt.clf()
        plt.figure()
        plot = sns.barplot(df_cancer_types, x="cancer_type_code", y="accuracy_score")
        plot.set_title("Accuracy for cancer types")
        plot.set(ylabel="Accuracy score", xlabel="Cancer type")
        plt.subplots_adjust(bottom=0.3)
        plt.xticks(rotation=90)
        save_fig(f"{save_to}/accuracy_cancer_type.svg", tight=False)

        plt.clf()

    @property
    def all_treatments(self) -> List[str]:
        """
        Return the dataframe of the dataset.
        """
        return list([x.lower() for x in self._all_treatments.keys()])

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
