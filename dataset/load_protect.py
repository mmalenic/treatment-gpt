import os
from ast import literal_eval
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import itertools

from classifer.util import find_file
from dataset.mutation_landscape_cancer_type import MutationLandscapeCancerType
from prepare.pubmed_downloader import PubmedDownloader


class LoadProtect:
    """
    Loads all protect data.
    """

    protect_ending = ".protect.tsv"
    random_state = 42

    def __init__(
        self,
        cancer_types: MutationLandscapeCancerType,
        sample_dir: str = "data/samples/",
        output_to: str = "data/load_protect.csv",
        pubmed_dir: str = "data/pubmed/",
        **kwargs
    ) -> None:
        """
        Initialize this class.

        :param sample_dir: sample directory.
        :param output_to: output to.
        :param cancer_types: list of cancer_types to load.
        """
        self.__dict__.update(kwargs)

        self._cancer_types = cancer_types
        self._sample_dir = sample_dir
        self._output_to = output_to
        self._pubmed_dir = pubmed_dir

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

    def sources(self) -> set[str]:
        """
        Get unique sources.
        """
        sources = list(self.df()["sources_x"]) + list(self.df()["sources_y"])
        return set(
            [
                x[0]
                for x in itertools.chain.from_iterable(
                    [literal_eval(x) for x in sources]
                )
                if x
            ]
        )

    def treatments_and_sources(self) -> set[Tuple[str, str, str]]:
        """
        Get unique sources.
        """
        sources = list(self.df()["treatment_with_text_sources_x"]) + list(
            self.df()["treatment_with_text_sources_y"]
        )
        return set([(y[0], y[1], y[2]) for x in sources for y in x])

    def load_pubmed(self) -> pd.DataFrame:
        def get_text_source(x, column):
            output = []
            for source in literal_eval(x[column]):
                treatment = source[1]

                out = None
                if len(treatment) != 0 and treatment is not None:
                    treatment = treatment[0]
                    treatment = find_file(self._pubmed_dir, treatment)
                    if treatment is not None:
                        out = (
                            source[0],
                            Path(treatment).read_text(encoding="utf-8"),
                            source[2],
                        )

                if treatment is None or len(treatment) == 0:
                    out = (source[0], "", source[2])

                output.append(out)

            return output

        if self._df is None:
            self.load()

        self.df()["treatment_with_text_sources_x"] = self.df().apply(
            lambda x: get_text_source(x, "treatment_with_source_and_level_x"), axis=1
        )
        self.df()["treatment_with_text_sources_y"] = self.df().apply(
            lambda x: get_text_source(x, "treatment_with_source_and_level_y"), axis=1
        )

        return self.df()

    def load(self) -> pd.DataFrame:
        """
        load the data.
        """

        def split_urls(x) -> List[str]:
            urls = str(x).split("||")[-1].split("|")[-1].split(",")

            output_urls = []
            for url in urls:
                url_match = PubmedDownloader.pubmed_url_regex.match(url)
                if url_match:
                    output_urls.append(url_match.group("pubmed_id"))

            return output_urls

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
                + x["treatment_with_source_and_level"]
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
                "treatment_with_source_and_level_x",
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
                "treatment_with_source_and_level_y",
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
                    "treatment_with_source_and_level_x",
                    "treatment_with_source_and_level_y",
                ]
            ]
            return output

        def p_val(x) -> pd.DataFrame:
            return next(
                y[2] for y in pairs if y[0] == x["gene_x"] and y[1] == x["gene_y"]
            )

        if os.path.exists(self._output_to):
            print("loading protect from:", self._output_to)
            self._df = pd.read_csv(self._output_to)
            self._stats = self._df.describe()

            return self._df

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

                frame["treatment_with_source_and_level"] = list(
                    zip(frame["treatment"], frame["sources"], frame["level"])
                )

                frame = frame[frame["onLabel"]]
                # frame = frame[(frame["level"] == "A") | (frame["level"] == "B") | (frame["level"] == "C")]
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
        output = output.sample(frac=1, random_state=self.random_state).reset_index(
            drop=True
        )
        output = output.drop_duplicates(subset=["cancer_type", "gene_x", "gene_y"])

        self._df = output
        self._stats = self._df.describe()

        print("saving protect to:", self._output_to)
        self._df.to_csv(self._output_to)

        return output
