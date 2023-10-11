import itertools
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.metrics import hamming_loss
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
        **kwargs
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

    def add_prediction(self, prediction, pos):
        self._df.iloc[pos, self._df.columns.get_loc("y_pred")] = prediction

    def dataset(self):
        """
        Return the dataset.
        """
        return self._dataset
