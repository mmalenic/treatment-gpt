import copy
import json
import os
import random
from hashlib import md5
from typing import Literal

import pandas as pd
from typing_extensions import override

from classifier.gpt.base_gpt_classifier import BaseGPTClassifier
from classifier.gpt.prompt_templates import *
from dataset.gene_pair_dataset import GenePairDataset
from dataset.load_protect import LoadProtect
from dataset.treatment_source_dataset import TreatmentSourceDataset


class TreatmentSourceGPTClassifier(BaseGPTClassifier):
    def __init__(
        self,
        base_dataset: TreatmentSourceDataset,
        prompt_template: Prompts.zero_shot_treatment_source_literal
        | Prompts.few_shot_treatment_source_literal
        | Prompts.zero_shot_treatment_source_cot_literal
        | Prompts.few_shot_treatment_source_cot_literal,
        model_type: Literal["gpt-3.5-turbo"]
        | Literal["gpt-3.5-turbo-16k"]
        | Literal["gpt-4"]
        | Literal["gpt-4-32k"] = "gpt-3.5-turbo",
        n_examples: int = 2,
        base_save_dir: str = "data/results",
        **kwargs,
    ):
        """
        Initialize this class.
        """

        if (
            prompt_template != Prompts.zero_shot_treatment_source_name
            and prompt_template != Prompts.few_shot_treatment_source_name
            and prompt_template != Prompts.zero_shot_treatment_source_cot_name
            and prompt_template != Prompts.few_shot_treatment_source_cot_name
        ):
            raise TypeError(
                f"Invalid prompt template for this classifier: {prompt_template}"
            )

        self.__dict__.update(kwargs)
        super().__init__(
            base_dataset.df,
            os.path.join(
                base_save_dir,
                (prompt_template + "_" + model_type)
                .replace(".", "_")
                .replace("-", "_"),
            ),
            model_type,
            **kwargs,
        )

        self.base_dataset = base_dataset
        self.prompt_template = Prompts.from_name(prompt_template)
        self.n_examples = n_examples

    @override
    def _construct_prompt(self, x) -> str:
        if "{examples}" in self.prompt_template:
            return self.prompt_template.format(
                treatments=json.dumps(x["treatments"]),
                source=x["source"],
                examples=self._construct_examples(x),
            )
        else:
            return self.prompt_template.format(
                treatments=json.dumps(x["treatments"]),
                source=x["source"],
            )

    def _construct_examples(self, x) -> str:
        dataset = copy.deepcopy(self.base_dataset.dataset())
        random.shuffle(dataset)

        examples = ""
        for i, treatment in enumerate([y for y in dataset if y["index"] != x["index"]]):
            if i == self.n_examples:
                break

            template = Prompts.treatment_source_prompt_template.format(
                source=treatment["source"],
                answer=json.dumps({"treatment": treatment["y_true"]}),
            )
            examples += template

        return examples

    @override
    def _index(self, x) -> str:
        return md5(f"{x['source']}_{x['y_true']}".encode("utf-8")).hexdigest()

    @override
    def _results(self, x) -> pd.DataFrame:
        return self.base_dataset.results(x)
