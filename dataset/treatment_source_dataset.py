import itertools

import numpy as np
import pandas as pd

from dataset.load_protect import LoadProtect
import random


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

    def load(self):
        """
        Load the dataset.
        """

        all_treatments = {}
        for index, row in itertools.islice(self._from_protect.df().iterrows(), 1):
            treatments = (
                row["treatment_with_text_sources_x"]
                + row["treatment_with_text_sources_y"]
            )
            for treatment in treatments:
                if treatment[1] is not None and treatment[1] != "":
                    all_treatments[(treatment[0], treatment[1])] = None

        for index, (treatment, source) in enumerate(all_treatments.keys()):
            if treatment is None or len(treatment) == 0:
                continue

            treatments = list(dict.fromkeys([x[0] for x in all_treatments]))
            if treatments and source is not None and source != "":
                self._dataset.append(
                    {
                        "index": index,
                        "source": source,
                        "treatments": treatments,
                        "y_true": treatment,
                        "y_pred": np.nan,
                        "loss": np.nan,
                    }
                )

        self._df = pd.DataFrame(self._dataset)

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
