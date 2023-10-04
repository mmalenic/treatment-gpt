import ast
import json
from json import JSONDecoder
from typing import Literal, List, Any, Dict, Optional
from abc import ABC, abstractmethod

import openai
from sklearn import metrics


class BaseGPTClassifier(ABC):
    def __init__(
        self,
        X,
        y,
        model_type: Literal["gpt-3.5-turbo"] | Literal["gpt-4"] = "gpt-3.5-turbo",
    ):
        """
        Initialize this class.
        """
        super().__init__()

        self._model_type = model_type
        self.X = X
        self.y = y
        self._predictions = None

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

    def predict(self):
        """Predict the labels."""
        self._predictions = [y for x in self._predict_single(self.X) for y in x]
        return self._predictions

    def _predict_single(self, x) -> List[str]:
        """Predict a single sample."""

        print("predicting for:", x)
        return []
        # response = openai.ChatCompletion.create(
        #     model=self._model_type,
        #     messages=[
        #         {"role": "system", "content": "You are a text classification model."},
        #         {"role": "user", "content": self._construct_prompt(x)},
        #     ],
        #     n=1,
        # )
        #
        # responses = []
        # for choice in response["choices"]:
        #     content = choice["message"]["content"]
        #     responses.append(self._extract_response(content, label=self._label()))
        #
        # return responses

    def _extract_response(self, content: str, label: str):
        """Extract the response."""

        def decode_dict(json_dict):
            try:
                results.append(json_dict[label])
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
    def _label(self) -> str:
        raise NotImplementedError
