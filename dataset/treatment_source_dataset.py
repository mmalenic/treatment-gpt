import numpy as np
import pandas as pd

from dataset.load_protect import LoadProtect
import random


class TreatmentSourceDataset:
    """
    Treatment source dataset
    """

    random_state = 42

    def __init__(self, from_protect: LoadProtect, **kwargs) -> None:
        """
        Initialize this class.

        :param from_protect: load protect data from this class.
        """
        self.__dict__.update(kwargs)

        self._dataset = []
        self._df = pd.DataFrame()
        self._from_protect = from_protect

        random.seed(self.random_state)

    def load(self):
        """
        Load the dataset.
        """

        all_treatments = set()
        for index, row in self._from_protect.df().iterrows():
            treatments = (
                row["treatment_with_text_sources_x"]
                + row["treatment_with_text_sources_y"]
            )
            for treatment in treatments:
                if treatment[1] is not None and treatment[1] != "":
                    all_treatments.add((treatment[0], treatment[1]))

        for index, (treatment, source) in enumerate(all_treatments):
            if treatment is None or len(treatment) == 0:
                continue

            treatments = list(set([x[0] for x in all_treatments]))
            if treatments and source is not None and source != "":
                self._dataset.append(
                    {
                        "index": index,
                        "source": source,
                        "treatments": treatments,
                        "y_true": treatment,
                        "y_pred": np.nan,
                    }
                )

        self._df = self.df()

    def df(self) -> pd.DataFrame:
        """
        Return the dataframe of the dataset.
        """
        return pd.DataFrame(self._dataset)

    def add_prediction(self, prediction, index):
        self._df.at[index, "y_pred"] = prediction

    def dataset(self):
        """
        Return the dataset.
        """
        return self._dataset
