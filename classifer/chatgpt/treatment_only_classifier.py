import copy
import random
from typing import Literal

import pandas as pd

from classifer.chatgpt.base_gpt_classifier import BaseGPTClassifier
from classifer.chatgpt.prompt_templates import *
from dataset.gene_pair_dataset import GenePairDataset
from dataset.load_protect import LoadProtect
from dataset.treatment_source_dataset import TreatmentSourceDataset


class TreatmentSourceGPTClassifier(BaseGPTClassifier):
    random_state = 42

    def __init__(
        self,
        base_dataset: TreatmentSourceDataset,
        prompt_template: str,
        model_type: Literal["gpt-3.5-turbo"] | Literal["gpt-4"] = "gpt-3.5-turbo",
        n_examples: int = 2,
    ):
        """
        Initialize this class.
        """

        y_true = [x["y_true"] for x in base_dataset.dataset()]
        super().__init__(base_dataset, y_true, model_type)

        self.base_dataset = base_dataset.dataset()
        self.prompt_template = prompt_template
        self.n_examples = n_examples

        random.seed(self.random_state)

    def _construct_prompt(self, x) -> str:
        if "{examples}" in self.prompt_template:
            return self.prompt_template.format(
                treatments=x["treatments"],
                source=x["source"],
                examples=self._construct_examples(x),
            )
        else:
            return self.prompt_template.format(
                treatments=x["treatments"],
                source=x["source"],
            )

    def _construct_examples(self, x) -> str:
        dataset = copy.deepcopy(self.base_dataset.dataset())
        random.shuffle(dataset)

        examples = ""
        for i, treatment in enumerate([y for y in dataset if y["index"] != x["index"]]):
            if i == self.n_examples:
                break

            template = TREATMENT_SOURCE_PROMPT_TEMPLATE.format(
                treatments=treatment["treatments"],
                source=treatment["source"],
            )
            examples += template

        return examples

    def _label(self) -> str:
        return "treatment"
