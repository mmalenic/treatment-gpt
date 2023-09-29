import os
from typing import List

import pandas as pd

from classifer.util import find_file
from dataset.mutation_landscape_cancer_type import MutationLandscapeCancerType


class LoadProtect:
    """
    Loads all protect data.
    """

    protect_ending = ".protect.tsv"

    def __init__(
        self,
        cancer_types: MutationLandscapeCancerType,
        sample_dir: str = "data/samples/",
        **kwargs
    ) -> None:
        """
        Initialize this class.

        :param sample_dir: sample directory.
        :param cancer_types: list of cancer_types to load.
        """
        self.__dict__.update(kwargs)

        self._cancer_types = cancer_types
        self._sample_dir = sample_dir

        self._df = None
        self._stats_after_concat = None
        self._stats_after_on_label_removed = None
        self._stats_after_level_removed = None

    def df(self) -> pd.DataFrame:
        """
        Get the data.
        """
        if self._df is None:
            self.load()

        return self._df

    @property
    def stats_after_concat(self) -> pd.DataFrame:
        """
        Get the stats after concat.
        """
        return self._stats_after_concat

    @property
    def stats_after_on_label_removed(self) -> pd.DataFrame:
        """
        Get the stats after onlabel removed.
        """
        return self._stats_after_on_label_removed

    @property
    def stats_after_level_removed(self) -> pd.DataFrame:
        """
        Get the stats after level removed.
        """
        return self._stats_after_level_removed

    def load(self) -> pd.DataFrame:
        """
        load the data.
        """
        dfs = {}
        for sample in os.listdir(self._sample_dir):
            sample_dir = os.path.join(self._sample_dir, sample)
            sample_id = os.path.basename(os.path.normpath(sample_dir))

            df = {}

            for protect_dir in os.listdir(sample_dir):
                protect_dir = os.path.join(sample_dir, protect_dir)
                protect_file = find_file(protect_dir, "*" + self.protect_ending)

                if (
                    protect_file is None
                    or protect_file == ""
                    or not os.path.exists(protect_file)
                ):
                    print("skipping loading protect for:", protect_dir)
                    continue

                print("loading protect for:", protect_dir)

                df[protect_dir] = pd.read_table(protect_file, sep="\t")

            try:
                dfs[sample_id] = pd.concat(df)
            except Exception as e:
                print("failed to load protect for:", sample_id, "with error:", e)
                continue

        print("number of samples:", len(dfs))

        output = pd.concat(dfs)
        self._stats_after_concat = output.describe()

        output = output[output["onLabel"]]
        self._stats_after_on_label_removed = output.describe()

        output = output[(output["level"] == "A") | (output["level"] == "B")]
        self._stats_after_level_removed = output.describe()

        self._df = output

        return output
