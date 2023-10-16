import itertools
from pathlib import Path
from typing import List

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from dataset.gene_pair_dataset import GenePairDataset
from dataset.utils import process_plus
from dataset.load_protect import LoadProtect
import random

from dataset.utils import (
    save_fig,
    heatmap_for_cls_report,
    results,
)


class TreatmentSourceDataset:
    """
    Treatment source dataset
    """

    def __init__(
        self, from_protect: LoadProtect, remove_generic_treatments=None, **kwargs
    ) -> None:
        """
        Initialize this class.

        :param from_protect: load protect data from this class.
        """
        self.__dict__.update(kwargs)

        self._dataset = []
        self._df = pd.DataFrame()
        self._from_protect = from_protect
        self._all_treatments = []

        if remove_generic_treatments is None:
            remove_generic_treatments = [
                "chemotherapy",
                "chemotherapy + everolimus + trastuzumab",
            ]

        self._remove_generic_treatments = remove_generic_treatments

    def load(self):
        """
        Load the dataset.
        """
        if not self._all_treatments:
            all_treatments = {}
            for index, row in self._from_protect.df().iterrows():
                treatments = (
                    row["treatment_with_text_sources_x"]
                    + row["treatment_with_text_sources_y"]
                )
                for treatment in treatments:
                    if treatment[1] is not None and treatment[1] != "":
                        all_treatments[
                            (
                                treatment[0],
                                treatment[1],
                                treatment[2],
                                row["cancer_type"],
                            )
                        ] = None

            self._all_treatments = list(all_treatments.keys())

        for index, (treatment, source, level, cancer_type) in enumerate(
            self._all_treatments
        ):
            if treatment is None or len(treatment) == 0:
                continue

            if (
                self._remove_generic_treatments
                and treatment.lower() in self._remove_generic_treatments
            ):
                continue

            treatments = list(dict.fromkeys([x[0] for x in self._all_treatments]))
            if treatments and source is not None and source != "":
                self._dataset.append(
                    {
                        "index": index,
                        "source": source,
                        "cancer_type_and_level": (cancer_type, level),
                        "treatments": treatments,
                        "y_true": process_plus(treatment, self),
                        "y_pred": np.nan,
                    }
                )

        self._df = pd.DataFrame(self._dataset)
        self._df = self._df.groupby(["source", "y_true"]).agg(list).reset_index()
        self._df["treatments"] = self._df["treatments"].apply(lambda x: x[0])
        self._df["y_pred"] = np.nan

    def diagrams(self, save_to: str):
        """
        Save all diagrams
        """

        def level_for_cancer_type(x):
            levels = []

            if not np.isnan(x["accuracy_a_level"]):
                levels.append("A")

            if not np.isnan(x["accuracy_b_level"]):
                levels.append("B")

            return ", ".join(levels)

        def heat_map(x, colour, save_to):
            return heatmap_for_cls_report(
                x,
                colour,
                save_to,
                x_label="Treatment",
                y_label="",
                title=f"Scores for treatment source dataset",
                width=30,
                height=10,
            )

        Path(save_to).mkdir(exist_ok=True, parents=True)

        df = self.df.copy()

        melt_level = df.melt(
            value_vars=["accuracy_a_level", "accuracy_b_level"],
        )
        melt_level = melt_level.rename(columns={"variable": "Evidence level"})
        melt_level.loc[
            melt_level["Evidence level"] == "accuracy_a_level",
            "Evidence level",
        ] = "A"
        melt_level.loc[
            melt_level["Evidence level"] == "accuracy_b_level",
            "Evidence level",
        ] = "B"

        melt_cancer_type = df.melt(
            id_vars=["accuracy_a_level", "accuracy_b_level"],
            value_vars=[f"accuracy_{cancer}" for cancer in self.all_cancer_types],
        )
        melt_cancer_type = melt_cancer_type.rename(columns={"variable": "Cancer type"})
        for cancer in self.all_cancer_types:
            melt_cancer_type.loc[
                melt_cancer_type["Cancer type"] == f"accuracy_{cancer}",
                "Cancer type",
            ] = self._from_protect.cancer_types.cancer_type_code(cancer)

        melt_cancer_type["Evidence level"] = melt_cancer_type.apply(
            level_for_cancer_type, axis=1
        )

        cls_report = pd.DataFrame(
            classification_report(
                self._df["y_true"].tolist(),
                GenePairDataset.process_predictions(self._df["y_pred"].tolist(), False),
                output_dict=True,
            )
        ).transpose()

        Path(f"{save_to}/heatmaps/").mkdir(exist_ok=True, parents=True)
        heat_map(cls_report, "Blues", f"{save_to}/heatmaps/heatmap_blue.svg")
        heat_map(cls_report, "Reds", f"{save_to}/heatmaps/heatmap_red.svg")
        heat_map(cls_report, "Greens", f"{save_to}/heatmaps/heatmap_green.svg")
        heat_map(cls_report, "Oranges", f"{save_to}/heatmaps/heatmap_orange.svg")
        heat_map(cls_report, "Purples", f"{save_to}/heatmaps/heatmap_purple.svg")

        plt.clf()
        plt.figure()
        plot = sns.barplot(melt_level, x="Evidence level", y="value")
        plot.set_title("Accuracy for evidence levels")
        plot.set(ylabel="Accuracy score", xlabel="Evidence level")
        save_fig(f"{save_to}/accuracy_level.svg")

        plt.clf()
        plt.figure()
        plot = sns.barplot(melt_cancer_type, x="Cancer type", y="value")
        plot.set_title("Accuracy for cancer types")
        plot.set(ylabel="Accuracy score", xlabel="Cancer types")
        plt.subplots_adjust(bottom=0.3)
        plt.xticks(rotation=90)
        save_fig(f"{save_to}/accuracy_cancer_type_level.svg")

        plt.clf()

    def results(self, x) -> pd.DataFrame:
        """
        Compute the results
        """
        try:
            x["accuracy_score"] = accuracy_score([x["y_true"]], [x["y_pred"]])
        except Exception as e:
            print("error:", e)
            x["accuracy_score"] = 0

        levels = [y[1] for y in x["cancer_type_and_level"]]
        cancer_types = [y[0] for y in x["cancer_type_and_level"]]

        if "A" in levels:
            x["accuracy_a_level"] = x["accuracy_score"]
        if "B" in levels:
            x["accuracy_b_level"] = x["accuracy_score"]

        for cancer in self.all_cancer_types:
            if cancer in cancer_types:
                x[f"accuracy_{cancer}"] = x["accuracy_score"]

        return x

    @property
    def all_cancer_types(self) -> List[str]:
        """
        Get all the cancer types for this dataset.
        """
        return list(dict.fromkeys([x[3] for x in self._all_treatments]))

    @property
    def all_treatments(self) -> List[str]:
        """
        Return the dataframe of the dataset.
        """
        return list(dict.fromkeys([x[0].lower() for x in self._all_treatments]))

    def get_all_treatments(self) -> List[str]:
        """
        Return the dataframe of the dataset.
        """
        return list(dict.fromkeys([x[0] for x in self._all_treatments]))

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

    def aggregate_results(self):
        """
        Get aggregate results.
        """
        y_true = self._df["y_true"].tolist()
        y_pred = GenePairDataset.process_predictions(self._df["y_pred"].tolist(), False)

        return results(y_true, y_pred, accuracy_score, sample_wise=False)

    def add_prediction(self, prediction, pos):
        self._df.iloc[pos, self._df.columns.get_loc("y_pred")] = prediction

    def dataset(self):
        """
        Return the dataset.
        """
        return self._dataset
