import ast
import json
import os
from json import JSONDecoder
from pathlib import Path
from typing import Literal, List, Any, Dict, Optional
from abc import ABC, abstractmethod
import tiktoken

import openai
from sklearn import metrics


class BaseGPTClassifier(ABC):
    output_n_tokens_estimate = 100

    def __init__(
        self,
        X,
        y,
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
        self.y = y
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.save_dir = save_dir
        self._predictions = None
        self._cost_estimate = None
        self._max_token_number = None

    def n_samples(self) -> int:
        return len(self.X)

    def loss(self) -> float | int:
        return metrics.hamming_loss(self.y, self._predictions)

    def confusion_matrix(self):
        return metrics.multilabel_confusion_matrix(
            self.y, self._predictions, samplewise=True
        )

    def f1_score(self):
        return metrics.f1_score(self.y, self._predictions, average="samples")

    @property
    def cost_estimate(self):
        """
        Estimate costs
        """
        estimates = []
        for x in self.X:
            n_tokens, estimate = self._cost_estimate_single(x)

            if n_tokens > self._max_token_number:
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
        self._predictions = [y for x in self.X for y in self._predict_single(x)]
        return self._predictions

    def _cost_estimate_single(self, x) -> (int, float):
        """
        Cost for a single sample.
        """
        encoding = tiktoken.encoding_for_model(self._model_type)
        n_tokens = len(encoding.encode(self._construct_prompt(x)))

        if self._model_type == "gpt-3.5-turbo":
            return n_tokens, (n_tokens * 0.0015) + (
                self.output_n_tokens_estimate * 0.002
            )
        elif self._model_type == "gpt-3.5-turbo-16k":
            return n_tokens, (n_tokens * 0.003) + (
                self.output_n_tokens_estimate * 0.004
            )
        elif self._model_type == "gpt-4":
            return n_tokens, (n_tokens * 0.03) + (self.output_n_tokens_estimate * 0.06)
        else:
            return n_tokens, (n_tokens * 0.06) + (self.output_n_tokens_estimate * 0.12)

    def _predict_single(self, x) -> List[str]:
        """
        Predict a single sample.
        """
        index = self._index(x)
        if Path(os.path.join(self.save_dir, index)).exists():
            with open(os.path.join(self.save_dir, index), "r", encoding="utf-8") as f:
                response = json.load(f)["response"]
        else:
            response = openai.ChatCompletion.create(
                model=self._model_type,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a text classification model.",
                    },
                    {"role": "user", "content": self._construct_prompt(x)},
                ],
                n=1,
            )
            with open(os.path.join(self.save_dir, index), "w", encoding="utf-8") as f:
                json.dump({"x": x, "response": response}, f)

        responses = []
        for choice in response["choices"]:
            content = choice["message"]["content"]
            responses.append(self._extract_response(content))

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
