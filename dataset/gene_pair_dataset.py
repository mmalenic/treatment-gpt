import numpy as np

from dataset.load_protect import LoadProtect
import random


class GenePairDataset:
    """
    Loads all protect data.
    """

    protect_ending = ".protect.tsv"
    random_state = 42

    def __init__(self, from_protect: LoadProtect, **kwargs) -> None:
        """
        Initialize this class.

        :param from_protect: load protect data from this class.
        """
        self.__dict__.update(kwargs)

        self._dataset = []
        self._additional_context = []
        self._from_protect = from_protect

        random.seed(self.random_state)

    def load(self):
        """Load the dataset."""
        for index, row in self._from_protect.df().iterrows():
            self._additional_context.append(row.tolist())

            treatments = (
                row["treatment_with_text_sources_x"]
                + row["treatment_with_text_sources_y"]
            )
            treatments = [(x[0], x[1]) for x in treatments]

            all_treatments = self._from_protect.treatments_and_sources()
            all_treatments = list([x for x in all_treatments if x[0] not in treatments])
            random.shuffle(all_treatments)

            treatments += all_treatments[: len(treatments)]

            y_true = [x for x, _, _ in treatments]

            self.dataset.append(
                {
                    "cancer_type": row["cancer_type"],
                    "gene_x": row["gene_x"],
                    "gene_y": row["gene_y"],
                    "treatments": treatments,
                    "y_true": y_true,
                }
            )
