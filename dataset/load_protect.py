import itertools
import os
from ast import literal_eval
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from classifier.util import find_file
from dataset.alternative_treatment_names import AlternativeTreatmentNames
from dataset.mutation_landscape_cancer_type import MutationLandscapeCancerType
from prepare.pubmed_downloader import PubmedDownloader


class LoadProtect:
    """
    Loads all protect data.
    """

    protect_ending = ".protect.tsv"

    def __init__(
        self,
        cancer_types: MutationLandscapeCancerType,
        alternative_names: AlternativeTreatmentNames,
        sample_dir: str = "data/samples/",
        output_to: str = "data/load_protect.csv",
        pubmed_dir: str = "data/pubmed/",
        gene_pairs_per_sample: bool = True,
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
        self._alternative_names = alternative_names
        self._sample_dir = sample_dir
        self._output_to = output_to
        self._gene_pairs_per_sample = gene_pairs_per_sample
        self._pubmed_dir = pubmed_dir

        self._df = None
        self._stats = None
        self._after_filtering = []
        self._total_protect_files = None
        self._total_samples = None
        self._total_empty_protect_results = 0

    @property
    def cancer_types(self) -> MutationLandscapeCancerType:
        """
        Get the data.
        """
        return self._cancer_types

    @property
    def alternative_names(self) -> AlternativeTreatmentNames:
        """
        Get the data.
        """
        return self._alternative_names

    def df(self) -> pd.DataFrame:
        """
        Get the data.
        """
        if self._df is None:
            self.load()

        return self._df.copy(deep=True)

    @property
    def stats(self) -> pd.DataFrame:
        """
        Get the stats.
        """
        return self._stats

    @property
    def after_filtering(self) -> List[int]:
        """
        Get the number of rows after filtering for each frame.
        """
        return self._after_filtering

    @property
    def total_protect_files(self) -> int:
        """
        Total protect files.
        """
        return self._total_protect_files

    @property
    def total_samples(self) -> int:
        """
        Total samples.
        """
        return self._total_samples

    @property
    def total_empty_protect_results(self) -> int:
        """
        Total empty protect results.
        """
        return self._total_empty_protect_results

    def sources(self) -> list[str]:
        """
        Get unique sources.
        """
        sources = list(self._df["sources_x"]) + list(self._df["sources_y"])
        return list(
            dict.fromkeys(
                [
                    x[0]
                    for x in itertools.chain.from_iterable(
                        [literal_eval(x) for x in sources]
                    )
                    if x
                ]
            )
        )

    def download_sources(self, email: str):
        """
        Download all sources.
        """

        sources = self.sources()
        pubmed = PubmedDownloader(email)

        for source in sources:
            pubmed.download(source)

    def treatments_and_sources(self) -> list[Tuple[str, str, str]]:
        """
        Get unique sources.
        """
        sources = list(self._df["treatment_with_text_sources_x"]) + list(
            self._df["treatment_with_text_sources_y"]
        )
        return list(dict.fromkeys([(y[0], y[1], y[2]) for x in sources for y in x]))

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

        self._df["treatment_with_text_sources_x"] = self._df.apply(
            lambda x: get_text_source(x, "treatment_with_source_and_level_x"), axis=1
        )
        self._df["treatment_with_text_sources_y"] = self._df.apply(
            lambda x: get_text_source(x, "treatment_with_source_and_level_y"), axis=1
        )

        return self._df

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

        def remove_duplicates(x) -> pd.DataFrame:
            x["sorted_gene_pairs"] = x.apply(
                lambda row: ";".join(sorted([row["gene_x"], row["gene_y"]])), axis=1
            )
            return x.drop_duplicates(subset=["cancer_type", "sorted_gene_pairs"])

        if os.path.exists(self._output_to):
            print("loading protect from:", self._output_to)
            self._df = pd.read_csv(self._output_to)
            self._stats = self._df.describe()

            return self._df

        dfs = {}
        for sample in os.listdir(self._sample_dir):
            sample_dir = os.path.join(self._sample_dir, sample)
            sample_id = os.path.basename(os.path.normpath(sample_dir))

            files = []
            samples = {}
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

                files.append((protect_file, protect_dir))
                samples[sample_dir] = None

            self._total_protect_files = len(files)
            self._total_samples = len(samples)

            df = []
            frames = []
            for protect_file, protect_dir in files:
                print("loading protect for:", protect_dir)

                frame = pd.read_table(protect_file, sep="\t")

                if frame.empty:
                    self._total_empty_protect_results += 1
                    continue

                cancer_type = self._cancer_type(protect_dir)
                frame["cancer_type"] = cancer_type

                frame["sources"] = frame["sources"].map(lambda x: split_urls(x))

                frame["treatment_with_source_and_level"] = list(
                    zip(frame["treatment"], frame["sources"], frame["level"])
                )

                frame = frame[frame["onLabel"]]
                frame = frame[(frame["level"] == "A") | (frame["level"] == "B")]
                frame = frame[frame["direction"] == "RESPONSIVE"]

                if frame.empty:
                    continue

                print("frame length after filtering:", frame.shape[0])
                self._after_filtering += [frame.shape[0]]

                frame["protect_dir"] = protect_dir

                frames.append(frame)

            if self._gene_pairs_per_sample:
                frames = self._gene_pairs(frames)

            for frame in frames:
                df.append(frame)

            try:
                dfs[sample_id] = pd.concat(df)
            except Exception as e:
                print("failed to load protect for:", sample_id, "with error:", e)
                continue

        print("number of samples:", len(dfs))

        output = pd.concat(dfs)
        output = output.sample(frac=1).reset_index(drop=True)

        if not self._gene_pairs_per_sample:
            output = pd.concat(self._gene_pairs([output]))

        output = remove_duplicates(output)

        self._df = output
        self._stats = self._df.describe()

        print("saving protect to:", self._output_to)
        self._df.to_csv(self._output_to)

        return output

    def _cancer_type(self, directory) -> str:
        doid = directory.split("_", 1)[1]
        return self._cancer_types.cancer_type(doid)

    def _gene_pairs(self, frames) -> List[pd.DataFrame]:
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
                + ";"
                + x["protect_dir"]
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
                "protect_dir_x",
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
                "protect_dir_y",
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
                    "protect_dir_x",
                    "protect_dir_y",
                ]
            ]
            return output

        def apply_types(x, cancer_type_pairs, pos):
            types = [
                y[pos]
                for y in cancer_type_pairs
                if (y[0] == x["gene_x"] and y[1] == x["gene_y"])
                or (y[1] == x["gene_x"] and y[0] == x["gene_y"])
            ]

            if len(types) != 2:
                raise Exception("incorrect p_vals found")

            return types[0]

        def match_with_cancer_types(x) -> pd.DataFrame:
            cancer_type = x["cancer_type"].iloc[0]
            cancer_type_pairs = pairs.loc[pairs["canonicalName"] == cancer_type][
                0
            ].tolist()[0]
            cancer_type_pairs += [
                (pair[1], pair[0], pair[2], pair[3], pair[4])
                for pair in cancer_type_pairs
            ]

            x = x[
                [
                    pair in [(p[0], p[1]) for p in cancer_type_pairs]
                    for pair in zip(x["gene_x"], x["gene_y"])
                ]
            ]

            if x.empty:
                return x

            x.loc[:, "p_val"] = x.apply(
                lambda y: apply_types(y, cancer_type_pairs, 2), axis=1
            )
            x["p_val"] = pd.to_numeric(x["p_val"])

            x.loc[:, "correlation_type"] = x.apply(
                lambda y: apply_types(y, cancer_type_pairs, 3), axis=1
            )

            x.loc[:, "odds"] = x.apply(
                lambda y: apply_types(y, cancer_type_pairs, 4), axis=1
            )
            x["odds"] = pd.to_numeric(x["odds"])

            return x

        out = []
        for frame in frames:
            frame = frame.groupby(["cancer_type", "gene"]).agg(list).reset_index()
            frame = (
                frame.groupby(["cancer_type"]).apply(gene_combinations).reset_index()
            )

            if frame.empty:
                continue

            pairs = (
                self._cancer_types.df()
                .groupby(["canonicalName"])[
                    ["genex", "geney", "pval", "corType", "odds"]
                ]
                .agg(list)
                .apply(
                    lambda x: list(
                        zip(x["genex"], x["geney"], x["pval"], x["corType"], x["odds"])
                    ),
                    axis=1,
                )
                .reset_index()
            )

            frame = (
                frame.groupby(["cancer_type"])
                .apply(match_with_cancer_types)
                .reset_index(drop=True)
            )

            if frame.empty:
                continue

            out.append(frame)

        return out
