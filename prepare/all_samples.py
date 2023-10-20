import os
import shutil

import boto3

from prepare.sample import Sample


class AllSamples:
    """
    Prepare all samples from the s3 bucket.
    """

    def __init__(
        self,
        prefix: str = "analysis_data/",
        output_dir: str = "data/samples/",
        bucket: str = "org.umccr.data.oncoanalyser",
    ) -> None:
        """
        Initialize this class.

        :param prefix: common prefix for samples.
        :param output_dir: output directory.
        :param bucket: bucket to use.
        """
        self._prefix = prefix
        self._output_dir = output_dir
        self._bucket = bucket
        self._samples = {}

    def prepare(self) -> None:
        """
        Prepare all samples from the bucket.
        """
        client = boto3.client("s3")
        result = client.list_objects_v2(
            Bucket=self._bucket, Prefix=self._prefix, Delimiter="/"
        )

        for obj in result.get("CommonPrefixes"):
            prefix = obj.get("Prefix")
            output_dir = os.path.join(
                self._output_dir, os.path.basename(os.path.normpath(prefix))
            )

            sample = Sample(prefix, output_dir, self._bucket)
            sample.prepare()

            if all(
                [
                    len(os.listdir(x[0])) == 0
                    for x in os.walk(output_dir)
                    if x[0] != output_dir
                ]
            ):
                shutil.rmtree(output_dir)

            self._samples[prefix] = sample
