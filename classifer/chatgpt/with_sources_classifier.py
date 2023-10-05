import copy
import json
import random
from hashlib import md5
from typing import Literal

import pandas as pd

from classifer.chatgpt.base_gpt_classifier import BaseGPTClassifier
from classifer.chatgpt.prompt_templates import *
from dataset.gene_pair_dataset import GenePairDataset
from dataset.load_protect import LoadProtect


class WithSourcesGenePairGPTClassifier(BaseGPTClassifier):
    random_state = 42

    def __init__(
        self,
        base_dataset: GenePairDataset,
        prompt_template: Prompts.zero_shot_with_sources_literal
        | Prompts.few_shot_with_sources_literal
        | Prompts.zero_shot_with_sources_cot_literal
        | Prompts.few_shot_with_sources_cot_literal,
        model_type: Literal["gpt-3.5-turbo"]
        | Literal["gpt-3.5-turbo-16k"]
        | Literal["gpt-4"]
        | Literal["gpt-4-32k"] = "gpt-3.5-turbo",
        n_examples: int = 2,
        **kwargs,
    ):
        """
        Initialize this class.
        """
        if (
            prompt_template != Prompts.zero_shot_with_sources_name
            and prompt_template != Prompts.few_shot_with_sources_name
            and prompt_template != Prompts.zero_shot_with_sources_cot_name
            and prompt_template != Prompts.few_shot_with_sources_cot_name
        ):
            raise TypeError(
                f"Invalid prompt template for this classifier: {prompt_template}"
            )

        y_true = [x["y_true"] for x in base_dataset.dataset()]

        self.__dict__.update(kwargs)
        super().__init__(
            base_dataset.dataset(), y_true, prompt_template, model_type, **kwargs
        )

        self.base_dataset = base_dataset
        self.prompt_template = Prompts.from_name(prompt_template)
        self.n_examples = n_examples

        random.seed(self.random_state)

    def _construct_prompt(self, x) -> str:
        treatments = self._construct_treatments_and_source(x)

        if "{examples}" in self.prompt_template:
            return self.prompt_template.format(
                treatments_and_sources=treatments,
                cancer_type=x["cancer_type"],
                genes=x["gene_x"] + " and " + x["gene_y"],
                examples=self._construct_examples(x),
                n_treatments=len(x["y_true"]),
            )
        else:
            return self.prompt_template.format(
                treatments_and_sources=treatments,
                n_treatments=len(x["y_true"]),
                cancer_type=x["cancer_type"],
                genes=x["gene_x"] + " and " + x["gene_y"],
            )

    @staticmethod
    def _construct_treatments_and_source(x) -> str:
        treatments_and_sources = ""

        for treatment in x["treatments"]:
            treatments_and_sources += (
                Prompts.treatment_and_source_prompt_template.format(
                    treatment=treatment["treatment"],
                    source=treatment["source"],
                )
            )

        return treatments_and_sources

    def _construct_examples(self, x) -> str:
        dataset = copy.deepcopy(self.base_dataset.dataset())
        random.shuffle(dataset)

        examples = ""
        genes = [x["gene_x"], x["gene_y"]]
        for i, treatment in enumerate(
            [
                y
                for y in dataset
                if y["index"] != x["index"]
                and y["gene_x"] not in genes
                and y["gene_y"] not in genes
            ]
        ):
            if i == self.n_examples:
                break

            template = Prompts.gene_pair_sources_example_prompt_template.format(
                treatments_and_sources=self._construct_treatments_and_source(treatment),
                cancer_type=treatment["cancer_type"],
                genes=treatment["gene_x"] + " and " + treatment["gene_y"],
                answer=json.dumps({"treatments": treatment["y_true"]}),
            )
            examples += template

        return examples

    def _index(self, x) -> str:
        return md5(
            f"{x['cancer_type']}_{x['gene_x']}_{x['gene_y']}_{x['y_true']}"
        ).hexdigest()
