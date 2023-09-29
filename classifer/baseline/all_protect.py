import json
import os
from typing import List, Optional

from classifer.baseline.linx import Linx
from classifer.baseline.protect import Protect
from classifer.util import find_file


class AllProtect:
    """
    Run protect for all samples
    """

    protect_ending = ".protect.tsv"

    def __init__(
        self,
        sample_dir: str = "data/samples/",
        doids: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """
        Initialize this class.

        :param sample_dir: sample directory.
        """
        self.__dict__.update(kwargs)

        if doids is None:
            doids = []

        self._sample_dir = sample_dir
        self._doids = doids
        self._runs = {}

    def run(self) -> None:
        """
        Run this class.
        """
        for sample in os.listdir(self._sample_dir):
            if "SBJ" not in sample:
                continue

            sample_dir = os.path.join(self._sample_dir, sample)

            print("running protect for:", sample_dir)

            try:
                linx = Linx(sample_dir)
                linx.run()
            except Exception as e:
                print(
                    "failed to run linx for: {0}, with error: {1}".format(sample_dir, e)
                )
                self._runs[sample_dir] = {"status": "failed", "errors": [str(e)]}

            protect_runs = []
            if not self._doids:
                protect_runs.append((Protect(sample_dir), sample_dir))
            else:
                for doid in self._doids:
                    protect_runs.append(
                        (
                            Protect(
                                sample_dir,
                                primary_tumor_doids=doid,
                                protect_directory="protect_" + doid + "/",
                            ),
                            sample_dir + "_" + doid,
                            os.path.join(sample_dir, "protect_" + doid + "/"),
                        )
                    )

            for protect, sample_dir, protect_directory in protect_runs:
                if os.path.exists(protect_directory) and find_file(
                    protect_directory, "*" + self.protect_ending
                ):
                    print("skipping running protect for:", sample, sample_dir)
                    self._runs[sample_dir] = {"status": "skipped", "errors": []}
                    continue

                try:
                    print("running protect for:", sample_dir)
                    protect.run()
                except Exception as e:
                    print(
                        "failed to run protect for: {0}, with error: {1}".format(
                            sample_dir, e
                        )
                    )
                    if sample_dir in self._runs:
                        self._runs[sample_dir]["errors"].append(str(e))
                    else:
                        self._runs[sample_dir] = {
                            "status": "failed",
                            "errors": [str(e)],
                        }

                if sample_dir not in self._runs:
                    self._runs[sample_dir] = {"status": "success", "errors": []}

        def check_status(x, status: str) -> bool:
            return isinstance(x, dict) and "status" in x and x["status"] == status

        self._runs["total_success"] = len(
            [x for x in self._runs.values() if check_status(x, "success")]
        )
        self._runs["total_failed"] = len(
            [x for x in self._runs.values() if check_status(x, "failed")]
        )
        self._runs["total_skipped"] = len(
            [x for x in self._runs.values() if check_status(x, "skipped")]
        )

        print(self._runs)

        with open(
            os.path.join(self._sample_dir, "protect_results.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(self._runs, f)
