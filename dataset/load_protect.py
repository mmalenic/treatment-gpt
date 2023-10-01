import os
from typing import List

import pandas as pd
import itertools

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
        self._stats = None

    def df(self) -> pd.DataFrame:
        """
        Get the data.
        """
        if self._df is None:
            self.load()

        return self._df

    @property
    def stats(self) -> pd.DataFrame:
        """
        Get the stats.
        """
        return self._stats

    def load(self) -> pd.DataFrame:
        """
        load the data.
        """

        def split_urls(x) -> List[str]:
            urls = str(x).split("||")[-1].split("|")[-1].split(",")
            return [
                url
                for url in urls
                if "pubmed.ncbi.nlm.nih.gov" in url or "ncbi.nlm.nih.gov/pubmed" in url
            ]

        def gene_combinations(x) -> pd.DataFrame:
            x = x.astype("str")

            x["temp"] = (
                x["gene"]
                + ";"
                + x["transcript"]
                + ";"
                + x["isCanonical"]
                + ";"
                + x["event"]
                + ";"
                + x["eventIsHighDriver"]
                + ";"
                + x["germline"]
                + ";"
                + x["reported"]
                + ";"
                + x["treatment"]
                + ";"
                + x["onLabel"]
                + ";"
                + x["level"]
                + ";"
                + x["direction"]
                + ";"
                + x["sources"]
                + ";"
                + x["cancer_type"]
                + ";"
                + x["treatment_with_source"]
            )

            new = pd.DataFrame([y for y in itertools.combinations(x.temp, 2)])
            if new.empty:
                return pd.DataFrame()

            new["temp"] = new[0] + ";" + new[1]

            output = new["temp"].str.split(";", expand=True)
            output.columns = [
                "gene_x",
                "transcript_x",
                "isCanonical_x",
                "event_x",
                "eventIsHighDriver_x",
                "germline_x",
                "reported_x",
                "treatment_x",
                "onLabel_x",
                "level_x",
                "direction_x",
                "sources_x",
                "cancer_type_x",
                "treatment_with_source_x",
                "gene_y",
                "transcript_y",
                "isCanonical_y",
                "event_y",
                "eventIsHighDriver_y",
                "germline_y",
                "reported_y",
                "treatment_y",
                "onLabel_y",
                "level_y",
                "direction_y",
                "sources_y",
                "cancer_type_y",
                "treatment_with_source_y",
            ]

            output = output[
                [
                    "gene_x",
                    "gene_y",
                    "transcript_x",
                    "transcript_y",
                    "isCanonical_x",
                    "isCanonical_y",
                    "event_x",
                    "event_y",
                    "eventIsHighDriver_x",
                    "eventIsHighDriver_y",
                    "germline_x",
                    "germline_y",
                    "reported_x",
                    "reported_y",
                    "treatment_x",
                    "treatment_y",
                    "onLabel_x",
                    "onLabel_y",
                    "level_x",
                    "level_y",
                    "direction_x",
                    "direction_y",
                    "sources_x",
                    "sources_y",
                    "cancer_type_x",
                    "cancer_type_y",
                    "treatment_with_source_x",
                    "treatment_with_source_y",
                ]
            ]
            return output

        def p_val(x) -> pd.DataFrame:
            return next(
                y[2] for y in pairs if y[0] == x["gene_x"] and y[1] == x["gene_y"]
            )

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

                frame = pd.read_table(protect_file, sep="\t")

                if frame.empty:
                    continue

                doid = protect_dir.split("_", 1)[1]
                cancer_type = self._cancer_types.cancer_type(doid)
                frame["cancer_type"] = cancer_type

                frame["sources"] = frame["sources"].map(lambda x: split_urls(x))

                frame["treatment_with_source"] = list(
                    zip(frame["treatment"], frame["sources"])
                )

                frame = frame[frame["onLabel"]]
                frame = frame[(frame["level"] == "A") | (frame["level"] == "B")]
                frame = frame[frame["direction"] == "RESPONSIVE"]

                if frame.empty:
                    continue

                frame = frame.groupby(["cancer_type", "gene"]).agg(list).reset_index()
                frame = (
                    frame.groupby(["cancer_type"])
                    .apply(gene_combinations)
                    .reset_index()
                )

                pairs = (
                    self._cancer_types.df()
                    .groupby(["canonicalName"])[["genex", "geney", "pval"]]
                    .agg(list)
                    .apply(
                        lambda x: list(zip(x["genex"], x["geney"], x["pval"])), axis=1
                    )
                    .reset_index()
                )
                pairs = pairs.loc[pairs["canonicalName"] == cancer_type][0].tolist()[0]
                pairs += [(pair[1], pair[0], pair[2]) for pair in pairs]

                if frame.empty:
                    continue

                frame = frame[
                    [
                        pair in [(p[0], p[1]) for p in pairs]
                        for pair in zip(frame["gene_x"], frame["gene_y"])
                    ]
                ]

                if frame.empty:
                    continue

                frame["p_val"] = frame.apply(lambda x: p_val(x), axis=1)

                df[protect_dir] = frame

            try:
                dfs[sample_id] = pd.concat(df)
            except Exception as e:
                print("failed to load protect for:", sample_id, "with error:", e)
                continue

        print("number of samples:", len(dfs))

        output = pd.concat(dfs)
        self._stats = output.describe()

        self._df = output

        return output
