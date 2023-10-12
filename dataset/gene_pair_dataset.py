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

        def save_fig(save_to, plt=plt):
            plt.tight_layout()
            plt.savefig(save_to, format="svg", dpi=600)
            try:
                plt.close()
            except AttributeError:
                pass

        def cls_report(x):
            return pd.DataFrame(
                classification_report(
                    self._binarizer.transform(x["y_true"].tolist()),
                    self._binarizer.transform(x["y_pred"].tolist()),
                    output_dict=True,
                    target_names=self._binarizer.classes_,
                )
            ).transpose()

        def heatmaps_per_cancer_type(x, colour, save_to):
            x = x.rename(columns={"treatment": "Treatment"})
            x["cancer_type"] = f"Scores for {x['cancer_type'].iloc[0]}"

            cmap = sns.color_palette(colour, as_cmap=True)
            x = x.set_index("Treatment")

            fig, ax = plt.subplots()
            fig.set_figheight(25)
            fig.set_figwidth(15)

            ax.set_xlabel("Treatment")
            ax.set_ylabel("Score")

            divider = make_axes_locatable(ax)
            cbar_ax = divider.new_horizontal(size="5%", pad=0.5, pack_start=False)
            fig.add_axes(cbar_ax)

            ax = sns.heatmap(
                x[["precision", "recall", "f1-score"]]
                .sort_values(by="f1-score")
                .transpose(),
                annot=True,
                linewidth=0.5,
                square=True,
                ax=ax,
                cbar_ax=cbar_ax,
                cmap=cmap,
                vmin=0,
                vmax=1,
                cbar_kws={"pad": 0.02},
            )
            ax.set_aspect("equal")

            ax.set_title(x["cancer_type"].iloc[0])

            save_fig(save_to, fig)

        Path(save_to).mkdir(exist_ok=True, parents=True)

        df = self.df.copy()
        df.loc[df["correlation_type"] == "none", "correlation_type"] = "no correlation"

        has_cor = df.loc[df["correlation_type"] != "no correlation"]
        mutually_exclusive = df.loc[df["correlation_type"] == "mutual exclusivity"]
        cooccuring = df.loc[df["correlation_type"] == "co-occurrence"]

        melt = df.melt(
            id_vars=["correlation_type"],
            value_vars=["accuracy_score_a_level", "accuracy_score_b_level"],
        )

        group_by_cancer_type = (
            df.groupby(["cancer_type"]).apply(cls_report).reset_index()
        )
        group_by_cancer_type = group_by_cancer_type.rename(
            columns={"level_1": "treatment"}
        )
        group_by_cancer_type = group_by_cancer_type[
            group_by_cancer_type.apply(
                lambda x: True
                if x["treatment"] in self.treatments_for(x["cancer_type"])
                else False,
                axis=1,
            )
        ]

        group_by_cancer_type.groupby(["cancer_type"]).apply(
            lambda x: heatmaps_per_cancer_type(
                x, "Blues", f"{save_to}/{x['cancer_type'].iloc[0]}_blue.svg"
            )
        )
        group_by_cancer_type.groupby(["cancer_type"]).apply(
            lambda x: heatmaps_per_cancer_type(
                x, "Reds", f"{save_to}/{x['cancer_type'].iloc[0]}_red.svg"
            )
        )
        group_by_cancer_type.groupby(["cancer_type"]).apply(
            lambda x: heatmaps_per_cancer_type(
                x, "Greens", f"{save_to}/{x['cancer_type'].iloc[0]}_green.svg"
            )
        )
        group_by_cancer_type.groupby(["cancer_type"]).apply(
            lambda x: heatmaps_per_cancer_type(
                x, "Oranges", f"{save_to}/{x['cancer_type'].iloc[0]}_orange.svg"
            )
        )
        group_by_cancer_type.groupby(["cancer_type"]).apply(
            lambda x: heatmaps_per_cancer_type(
                x, "Purples", f"{save_to}/{x['cancer_type'].iloc[0]}_purple.svg"
            )
        )

        plt.figure()
        plot = sns.barplot(df, x="correlation_type", y="accuracy_score").set_title(
            "Accuracy for correlation type"
        )
        plot.label(x="Correlation type", y="Accuracy score")
        save_fig(f"{save_to}/accuracy_score.svg")

        plt.figure()
        plot = sns.lmplot(
            data=has_cor, x="accuracy_score", y="p_val", hue="correlation_type"
        ).set_title("Accuracy and p-value for correlations")
        plot.label(x="Accuracy score", y="p-value", color="Correlation type")
        save_fig(f"{save_to}/accuracy_p_val_for_correlation.svg")

        plt.figure()
        plot = sns.lmplot(
            data=has_cor, x="accuracy_score", y="odds", hue="correlation_type"
        ).set_title("Accuracy and odds for correlations")
        plot.label(x="Accuracy score", y="Odds", color="Correlation type")
        save_fig(f"{save_to}/accuracy_odds_for_correlation.svg")

        plt.figure()
        plot = sns.lmplot(
            data=mutually_exclusive,
            x="accuracy_score",
            y="p_val",
            hue="correlation_type",
        ).set_title("Accuracy and p-value for mutually exclusive correlations")
        plot.label(x="Accuracy score", y="p-value", color="Correlation type")
        save_fig(f"{save_to}/accuracy_p_value_mutually_exclusive.svg")

        plt.figure()
        plot = sns.lmplot(
            data=mutually_exclusive,
            x="accuracy_score",
            y="odds",
            hue="correlation_type",
        ).set_title("Accuracy and odds for mutually exclusive correlations")
        plot.label(x="Accuracy score", y="Odds", color="Correlation type")
        save_fig(f"{save_to}/accuracy_odds_mutually_exclusive.svg")

        plt.figure()
        plot = sns.lmplot(
            data=cooccuring, x="accuracy_score", y="p_val", hue="correlation_type"
        ).set_title("Accuracy and p-value for co-occurring correlations")
        plot.label(x="Accuracy score", y="p-value", color="Correlation type")
        save_fig(f"{save_to}/accuracy_p_value_co_occurring.svg")

        plt.figure()
        plot = sns.lmplot(
            data=cooccuring, x="accuracy_score", y="odds", hue="correlation_type"
        ).set_title("Accuracy and odds for co-occurring correlations")
        plot.label(x="Accuracy score", y="Odds", color="Correlation type")
        save_fig(f"{save_to}/accuracy_odds_co_occurring.svg")

        plt.figure()
        plot = sns.barplot(
            melt, x="correlation_type", y="value", hue="variable"
        ).set_title("Accuracy for evidence levels and correlation type")
        plot.label(x="Accuracy score", y="Correlation type", color="Evidence level")

        labels = ["A", "B"]
        for text, label in zip(plot._legend.texts, labels):
            text.set_text(label)

        save_fig(f"{save_to}/accuracy_level_correlation_type.svg")

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

    def add_prediction(self, prediction, pos):
        self._df.iloc[pos, self._df.columns.get_loc("y_pred")] = prediction

    def dataset(self):
        """
        Return the dataset.
        """
        return self._dataset
