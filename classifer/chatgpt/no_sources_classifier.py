from typing import Literal

import pandas as pd

from classifer.chatgpt.base_gpt_classifier import BaseGPTClassifier
from classifer.chatgpt.prompt_templates import *
from dataset.load_protect import LoadProtect


class NoSourcesGPTClassifier(BaseGPTClassifier):
    def __init__(
        self,
        base_dataset: LoadProtect,
        prompt_template: str,
        model_type: Literal["gpt-3.5-turbo"] | Literal["gpt-4"] = "gpt-3.5-turbo",
    ):
        """Initialize this class."""

        super().__init__(model_type)

        self.base_dataset = base_dataset
        self.prompt_template = prompt_template

    @staticmethod
    def construct_dataset(base_dataset: pd.DataFrame):
        dataset = []
        for row in base_dataset.iterrows():
            dataset.append(row["gene_x"], row["gene_y"], row["treatment"])
            gene_x = row["gene_x"]
            gene_y = row["gene_y"]

    def _construct_prompt(self, x) -> str:
        pass
        # prompt = self.prompt_template.format(
        #     treatments=
        # )

    def _label(self) -> str:
        return "treatments"
