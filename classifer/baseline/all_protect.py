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
                self._runs[sample_dir] = {"status": "failed", "errors": [e]}

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
                    self._runs[sample_dir]["errors"].append(e)
                else:
                    self._runs[sample_dir] = {"status": "failed", "errors": [e]}

            if sample_dir not in self._runs:
                self._runs[sample_dir] = {"status": "success", "errors": []}
