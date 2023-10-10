import ast
import json
import os
from json import JSONDecoder
from pathlib import Path
from typing import Literal, List, Any, Dict, Optional
from abc import ABC, abstractmethod

import pandas as pd
import tiktoken
from decimal import *

import openai
from sklearn import metrics
from sklearn.metrics import hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer

from classifier.gpt.prompt_templates import Prompts


class BaseGPTClassifier(ABC):
    output_n_tokens_estimate = 200
    output_n_tokens_estimate_cot = 1000
    max_4k_input = 4097 - output_n_tokens_estimate
    max_4k_input_cot = 4097 - output_n_tokens_estimate_cot
    max_8k_input = 8192 - output_n_tokens_estimate
    max_8k_input_cot = 8192 - output_n_tokens_estimate_cot

    def __init__(
        self,
        X: pd.DataFrame,
        save_dir: str,
        model_type: Literal["gpt-3.5-turbo"]
        | Literal["gpt-3.5-turbo-16k"]
        | Literal["gpt-4"]
        | Literal["gpt-4-32k"] = "gpt-3.5-turbo",
        **kwargs,
    ):
        """
        Initialize this class.
        """
        super().__init__()
        self.__dict__.update(kwargs)

        self._model_type = model_type
        self.X = X
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.save_dir = save_dir
        self._cost_estimate = None
        self._max_token_number = None

    def n_samples(self) -> int:
        return len(self.X)

    @property
    def cost_estimate(self):
        """
        Estimate costs
        """
        estimates = []
        for x in self.X:
            prompt = self._construct_prompt(x)
            n_tokens = self._n_tokens(prompt)
            model_type = self._get_model_type(n_tokens, prompt)
            estimate = self._cost_estimate_single(model_type, n_tokens)

            if self._max_token_number is None or n_tokens > self._max_token_number:
                self._max_token_number = n_tokens

            estimates.append(estimate)

        self._cost_estimate = sum(estimates)

        return self._cost_estimate

    @property
    def max_token_number(self):
        """
        Maximum number of tokens in the dataset
        """
        return self._max_token_number

    def predict(self):
        """
        Predict the labels.
        """
        self.X.apply(lambda x: self._predict_single(x), axis=1)

    def _n_tokens(self, prompt) -> int:
        """
        Get the number of tokens in a prompt.
        """
        encoding = tiktoken.encoding_for_model(self._model_type)
        return len(encoding.encode(prompt))

    def _cost_estimate_single(self, model_type: str, n_tokens: int) -> Decimal:
        """
        Cost for a single sample.
        """
        if model_type == "gpt-3.5-turbo":
            return ((Decimal(n_tokens) / Decimal(1000)) * Decimal(0.0015)) + (
                (Decimal(self.output_n_tokens_estimate) / Decimal(1000))
                * Decimal(0.002)
            )
        elif model_type == "gpt-3.5-turbo-16k":
            return ((Decimal(n_tokens) / Decimal(1000)) * Decimal(0.003)) + (
                (Decimal(self.output_n_tokens_estimate) / Decimal(1000))
                * Decimal(0.004)
            )
        elif model_type == "gpt-4":
            return ((Decimal(n_tokens) / Decimal(1000)) * Decimal(0.03)) + (
                (Decimal(self.output_n_tokens_estimate) / Decimal(1000)) * Decimal(0.06)
            )
        else:
            return ((Decimal(n_tokens) / Decimal(1000)) * Decimal(0.06)) + (
                (Decimal(self.output_n_tokens_estimate) / Decimal(1000)) * Decimal(0.12)
            )

    def _get_model_type(self, n_tokens: int, prompt: str) -> str:
        """
        Get the best model type based on the number of tokens.
        """
        model_type = self._model_type

        if (
            Prompts.cot_prompt_template in prompt
            or Prompts.treatment_only_cot_prompt_template in prompt
        ):
            if model_type == "gpt-3.5-turbo" and n_tokens > self.max_4k_input_cot:
                model_type = "gpt-3.5-turbo-16k"
            if model_type == "gpt-4" and n_tokens > self.max_8k_input_cot:
                model_type = "gpt-4-32k"

        if (
            Prompts.cot_prompt_template not in prompt
            and Prompts.treatment_only_cot_prompt_template not in prompt
        ):
            if model_type == "gpt-3.5-turbo" and n_tokens > self.max_4k_input:
                model_type = "gpt-3.5-turbo-16k"
            if model_type == "gpt-4" and n_tokens > self.max_8k_input:
                model_type = "gpt-4-32k"

        return model_type

    def _predict_single(self, x) -> List[str]:
        """
        Predict a single sample.
        """
        index = self._index(x)
        if Path(os.path.join(self.save_dir, index)).exists():
            with open(os.path.join(self.save_dir, index), "r", encoding="utf-8") as f:
                response = json.load(f)["response"]
        else:
            prompt = self._construct_prompt(x)
            n_tokens = self._n_tokens(prompt)
            model_type = self._get_model_type(n_tokens, prompt)

            response = openai.ChatCompletion.create(
                model=model_type,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a text classification model.",
                    },
                    {"role": "user", "content": prompt},
                ],
                n=1,
            )

            for choice in response["choices"]:
                if choice["finish_reason"] == "length":
                    raise ValueError("Model returned a truncated response.")

            with open(os.path.join(self.save_dir, index), "w", encoding="utf-8") as f:
                json.dump({"x": x.to_json(), "response": response}, f)

        responses = []
        for choice in response["choices"]:
            content = choice["message"]["content"]
            responses.append(self._extract_response(content))

        y_true = [x["y_true"]] * len(responses)

        binarizer = MultiLabelBinarizer()
        binarizer.fit(y_true)

        y_true = binarizer.transform(y_true)
        y_pred = binarizer.transform(responses)

        x["loss"] = hamming_loss(y_true, y_pred)

        return responses

    def _extract_response(self, content: str):
        """
        Extract the response.
        """

        def decode_dict(json_dict):
            try:
                results.append(json_dict["treatment"])
            except KeyError:
                pass
            try:
                results.append(json_dict["treatments"])
            except KeyError:
                pass
            return json_dict

        json_content = self._extract_json_objects(content)
        if len(json_content) > 1:
            raise ValueError("More than one JSON object found.")
        if len(json_content) == 0:
            raise ValueError("No JSON object found.")
        json_content = json_content[0]

        results = []
        json.loads(json.dumps(json_content), object_hook=decode_dict)

        if len(results) > 1:
            raise ValueError("More than one label found.")
        if len(results) == 0:
            raise ValueError("No label found.")

        return results[0]

    @staticmethod
    def _extract_json_objects(text, decoder=JSONDecoder()) -> List[Any]:
        """
        Find JSON objects in text, and yield the decoded JSON data
        """

        pos = 0
        results = []
        while True:
            match = text.find("{", pos)
            if match == -1:
                break
            try:
                result, index = decoder.raw_decode(text[match:])
                results.append(result)
                pos = match + index
            except ValueError:
                pos = match + 1

        return results

    @abstractmethod
    def _construct_prompt(self, x) -> str:
        raise NotImplementedError

    @abstractmethod
    def _index(self, x) -> str:
        raise NotImplementedError
