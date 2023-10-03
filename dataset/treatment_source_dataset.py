import numpy as np

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
                all_treatments.add((treatment[0], treatment[1]))

        for treatment, source in all_treatments:
            self._dataset.append(
                {
                    "source": source,
                    "treatments": list(set([x[0] for x in all_treatments])),
                    "y_true": treatment,
                }
            )

    def dataset(self):
        """
        Return the dataset.
        """
        return self._dataset
