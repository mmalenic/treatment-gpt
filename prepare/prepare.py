from typing import Any

from prepare.all_samples import AllSamples
from prepare.reference import Reference
from prepare.serve import Serve
from prepare.software import Software


class Prepare:
    """
    Prepare all samples from the s3 bucket.
    """

    def __init__(
        self,
        **kwargs,
    ) -> None:
        """
        Initialize this class.
        """
        if "software" in kwargs:
            self._software = Software(**kwargs["software"])
        else:
            self._software = Software()

        if "serve" in kwargs:
            self._serve = Serve(**kwargs["serve"])
        else:
            self._serve = Serve()

        if "reference" in kwargs:
            self._reference = Reference(**kwargs["reference"])
        else:
            self._reference = Reference()

        if "all_samples" in kwargs:
            self._all_samples = AllSamples(**kwargs["all_samples"])
        else:
            self._all_samples = AllSamples()

    def prepare(self) -> None:
        """
        Prepare all required components.
        """
        print("running software prepare")
        self._software.prepare()

        print("running serve prepare")
        self._serve.prepare()

        print("running reference prepare")
        self._reference.prepare()

        print("running all samles prepare")
        self._all_samples.prepare()
