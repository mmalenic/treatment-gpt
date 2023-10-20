import copy
import json
import os
import random
from hashlib import md5

import pandas as pd

from classifier.gpt.base_gpt_classifier import BaseGPTClassifier
from classifier.gpt.prompt_templates import *
from dataset.treatment_source_dataset import TreatmentSourceDataset


class TreatmentSourceNoListGPTClassifier(BaseGPTClassifier):
    def __init__(
        self,
        base_dataset: TreatmentSourceDataset,
        prompt_template: Prompts.zero_shot_treatment_source_no_list_literal
        | Prompts.few_shot_treatment_source_no_list_literal
        | Prompts.zero_shot_treatment_source_no_list_cot_literal
        | Prompts.few_shot_treatment_source_no_list_cot_literal,
        model_type: Literal["gpt-3.5-turbo"]
        | Literal["gpt-3.5-turbo-16k"]
        | Literal["gpt-4"]
        | Literal["gpt-4-32k"] = "gpt-3.5-turbo",
        n_examples: int = 2,
        repeat_n_times: int = 1,
        base_save_dir: str = "data/results",
        **kwargs,
    ):
        """
        Initialize this class.
        """

        if (
            prompt_template != Prompts.zero_shot_treatment_source_no_list_name
            and prompt_template != Prompts.few_shot_treatment_source_no_list_name
            and prompt_template != Prompts.zero_shot_treatment_source_no_list_cot_name
            and prompt_template != Prompts.few_shot_treatment_source_no_list_cot_name
        ):
            raise TypeError(
                f"Invalid prompt template for this classifier: {prompt_template}"
            )

        self.__dict__.update(kwargs)
        super().__init__(
            base_dataset.df,
            base_dataset,
            os.path.join(
                base_save_dir,
                (prompt_template + "_" + model_type)
                .replace(".", "_")
                .replace("-", "_"),
            ),
            model_type,
            repeat_n_times=repeat_n_times,
            **kwargs,
        )

        self.base_dataset = base_dataset
        self.prompt_template = Prompts.from_name(prompt_template)
        self.n_examples = n_examples

    def _construct_prompt(self, x) -> str:
        if "{examples}" in self.prompt_template:
            return self.prompt_template.format(
                source=x["source"],
                examples=self._construct_examples(x),
            )
        else:
            return self.prompt_template.format(
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

    def _index(self, x) -> str:
        return md5(f"{x['source']}_{x['y_true']}".encode("utf-8")).hexdigest()

    def _results(self, x) -> pd.DataFrame:
        return self.base_dataset.results(x)
