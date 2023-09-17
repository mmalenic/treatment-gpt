import os
import boto3
from prepare.sample import Sample


class AllSamples:
    """
    Prepare all samples from the s3 bucket.
    """

    _samples: dict[str, Sample] = {}

    def __init__(
        self,
        output_dir: str,
        bucket: str = "org.umccr.data.oncoanalyser",
    ) -> None:
        self._output_dir = output_dir
        self._bucket = bucket

    def prepare(self) -> None:
        """
        Prepare all samples from the bucket.
        """
        client = boto3.client("s3")
        result = client.list_objects(Bucket=self._bucket, Delimiter="/")

        for obj in result.get("CommonPrefixes"):
            prefix = obj.get("Prefix")
            output_dir = os.path.join(self._output_dir, prefix)

            sample = Sample(prefix, output_dir, self._bucket)
            sample.prepare()

            self._samples[prefix] = sample
