import sys
import random

import numpy as np
import openai

from classifier.gpt.run_configuration import RunConfiguration
from dataset.alternative_treatment_names import AlternativeTreatmentNames
from dataset.gene_pair_dataset import GenePairDataset
from dataset.load_protect import LoadProtect
from dataset.treatment_source_dataset import TreatmentSourceDataset

sys.path.append("..")
from classifier.baseline.all_protect import AllProtect
from dataset.mutation_landscape_cancer_type import MutationLandscapeCancerType
from prepare.all_samples import AllSamples
import seaborn as sns
import matplotlib.pyplot as plt


class RunClassifier:
    def __init__(self):
        """
        Initialize this class.
        """

        def load_dataset(dataset):
            dataset.load()
            return dataset

        self._mutation_landscape = MutationLandscapeCancerType()
        self._mutation_landscape.load()
        self._mutation_landscape.doids()

        self._names = AlternativeTreatmentNames()
        self._names.load()

        self._load = LoadProtect(
            cancer_types=self._mutation_landscape,
            gene_pairs_per_sample=False,
            alternative_names=self._names,
        )
        self._load.load()
        self._load.sources()
        self._load.load_pubmed()

        self._config = RunConfiguration(
            lambda: load_dataset(
                GenePairDataset(self._load, split_to_n_treatments=None)
            ),
            lambda: load_dataset(TreatmentSourceDataset(self._load)),
        )

    async def run(self):
        """
        Run the classifier
        """
        await self._config.run_all()
