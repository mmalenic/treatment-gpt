import os
import pandas as pd
import xarray as xr

from classifer.util import find_file


class LoadProtect:
    """
    Loads all protect data.
    """

    protect_directory = "protect/"
    protect_ending = ".protect.tsv"

    def __init__(self, sample_dir: str = "data/samples/", **kwargs) -> None:
        """
        Initialize this class.

        :param sample_dir: sample directory.
        """
        self.__dict__.update(kwargs)

        self._sample_dir = sample_dir

    def load(self) -> pd.DataFrame:
        """
        load the data.
        """
        dfs = {}
        for sample in os.listdir(self._sample_dir):
            sample_dir = os.path.join(self._sample_dir, sample)
            sample_id = os.path.basename(os.path.normpath(sample_dir))

            protect_dir = os.path.join(sample_dir, self.protect_directory)
            protect_file = find_file(protect_dir, "*" + self.protect_ending)

            if (
                protect_file is None
                or protect_file == ""
                or not os.path.exists(protect_file)
            ):
                print("skipping loading protect for:", protect_dir)
                continue

            print("loading protect for:", protect_dir)

            dfs[sample_id] = pd.read_table(protect_file, sep="\t")

        return pd.concat(dfs, ignore_index=True)
