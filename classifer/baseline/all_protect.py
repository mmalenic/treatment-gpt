import json
import os

from classifer.baseline.linx import Linx
from classifer.baseline.protect import Protect


class AllProtect:
    """
    Run protect for all samples
    """

    def __init__(
        self,
        sample_dir: str = "data/samples/",
    ) -> None:
        """
        Initialize this class.

        :param sample_dir: sample directory.
        """
        self._sample_dir = sample_dir
        self._runs = {}

    def run(self) -> None:
        """
        Run this class.
        """
        for sample in os.listdir(self._sample_dir):
            sample_dir = os.path.join(self._sample_dir, sample)

            # if os.path.exists(os.path.join(sample_dir, "protect")):
            #     print("skipping running protect for:", sample_dir)
            #     self._runs[sample_dir] = {"status": "skipped", "errors": []}
            #     continue

            print("running protect for:", sample_dir)

            try:
                linx = Linx(sample_dir)
                linx.run()
            except Exception as e:
                print(
                    "failed to run linx for: {0}, with error: {1}".format(sample_dir, e)
                )
                self._runs[sample_dir] = {"status": "failed", "errors": [str(e)]}

            try:
                protect = Protect(sample_dir)
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
                    self._runs[sample_dir] = {"status": "failed", "errors": [str(e)]}

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
