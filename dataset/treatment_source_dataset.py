import itertools
from pathlib import Path
from typing import List

import matplotlib as plt
import seaborn as sns
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import accuracy_score, classification_report

from dataset.load_protect import LoadProtect
import random

from dataset.diagram_utils import (
    save_fig,
    add_cancer_types_code,
    heatmaps_per_cancer_type,
    plot_heatmaps,
)


class TreatmentSourceDataset:
    """
    Treatment source dataset
    """

    def __init__(self, from_protect: LoadProtect, **kwargs) -> None:
        """
        Initialize this class.

        :param from_protect: load protect data from this class.
        """
        self.__dict__.update(kwargs)

        self._dataset = []
        self._df = pd.DataFrame()
        self._from_protect = from_protect
        self._all_treatments = []

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

            treatments = list(
                dict.fromkeys([x[0].lower() for x in self._all_treatments])
            )
            if treatments and source is not None and source != "":
                self._dataset.append(
                    {
                        "index": index,
                        "source": source,
                        "level": level,
                        "cancer_type": cancer_type,
                        "treatments": treatments,
                        "y_true": treatment.lower(),
                        "y_pred": np.nan,
                    }
                )

        self._df = pd.DataFrame(self._dataset)

    def diagrams(self, save_to: str):
        """
        Save all diagrams
        """

        def cls_report(x):
            return pd.DataFrame(
                classification_report(
                    x["y_true"].tolist(),
                    x["y_pred"].tolist(),
                    output_dict=True,
                )
            ).transpose()

        Path(save_to).mkdir(exist_ok=True, parents=True)
        Path(f"{save_to}/heatmaps/").mkdir(exist_ok=True, parents=True)

        df = self.df.copy()

        df_cancer_types = df.apply(
            lambda x: add_cancer_types_code(x, self._from_protect), axis=1
        )

        group_by_cancer_type = (
            df.groupby(["cancer_type"]).apply(cls_report).reset_index()
        )
        group_by_cancer_type = group_by_cancer_type.rename(
            columns={"level_1": "Treatment"}
        )

        plot_heatmaps(group_by_cancer_type, save_to)

        plt.clf()
        plt.figure()
        plot = sns.barplot(df_cancer_types, x="cancer_type_code", y="accuracy_score")
        plot.set_title("Accuracy for cancer types")
        plot.set(ylabel="Accuracy score", xlabel="Cancer type")
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.3)
        save_fig(f"{save_to}/accuracy_cancer_type.svg", tight=False)

        plt.clf()
        plt.figure()
        plot = sns.barplot(df_cancer_types, x="level", y="accuracy_score")
        plot.set_title("Accuracy for cancer types")
        plot.set(ylabel="Accuracy score", xlabel="Evidence level")
        save_fig(f"{save_to}/accuracy_cancer_type.svg", tight=False)

        plt.clf()

    def results(self, x) -> pd.DataFrame:
        """
        Compute the results
        """
        x["accuracy_score"] = accuracy_score(x["y_true"], x["y_pred"])
        return x

    @property
    def all_treatments(self) -> List[str]:
        """
        Return the dataframe of the dataset.
        """
        return list(dict.fromkeys([x[0].lower() for x in self._all_treatments]))

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
