import numpy as np

from dataset.load_protect import LoadProtect
import random


class GenePairDataset:
    """
    Loads all protect data.
    """

    random_state = 42

    def __init__(
        self, from_protect: LoadProtect, n_treatment_shuffles: int = 3, **kwargs
    ) -> None:
        """
        Initialize this class.

        :param from_protect: load protect data from this class.
        """
        self.__dict__.update(kwargs)

        self._dataset = []
        self._from_protect = from_protect
        self._n_treatment_shuffles = n_treatment_shuffles

        random.seed(self.random_state)

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
                {"treatment": x[0], "source": x[1], "level": x[2]} for x in treatments
            ]

            y_true = [x for x, _, _ in treatments]

            all_treatments = self._from_protect.treatments_and_sources()
            all_treatments = list([x for x in all_treatments if x[0] not in treatments])

            for _ in range(self._n_treatment_shuffles):
                random.shuffle(all_treatments)
                treatments += all_treatments[: len(treatments)]

                self._dataset.append(
                    {
                        "index": index,
                        "cancer_type": row["cancer_type"],
                        "gene_x": row["gene_x"],
                        "gene_y": row["gene_y"],
                        "treatments": treatments,
                        "y_true": y_true,
                    }
                )

    def dataset(self):
        """
        Return the dataset.
        """
        return self._dataset
