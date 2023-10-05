import numpy as np

from dataset.load_protect import LoadProtect
import random


class GenePairDataset:
    """
    Loads all protect data.
    """

    random_state = 42

    def __init__(
        self, from_protect: LoadProtect, remove_empty_sources: bool = False, **kwargs
    ) -> None:
        """
        Initialize this class.

        :param from_protect: load protect data from this class.
        """
        self.__dict__.update(kwargs)

        self._dataset = []
        self._from_protect = from_protect
        self._remove_empty_sources = remove_empty_sources

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

            y_true = [x["treatment"] for x in treatments]

            if len(y_true) == 0 or y_true is None:
                continue

            all_treatments = self._from_protect.treatments_and_sources()
            all_treatments = list([x for x in all_treatments if x[0] not in y_true])

            if self._remove_empty_sources:
                all_treatments = [
                    x for x in all_treatments if x[1] is not None and x[1] != ""
                ]

            random.shuffle(all_treatments)
            treatments += [
                {"treatment": x[0], "source": x[1], "level": x[2]}
                for x in all_treatments[: len(y_true)]
            ]

            random.shuffle(treatments)
            if treatments and y_true:
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
