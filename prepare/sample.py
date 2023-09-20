import os

from prepare.downloader import Downloader


class Sample:
    """
    Prepare sample data for running PROTECT
    """

    chord_directory = "chord/"
    lilac_directory = "lilac/"
    purple_directory = "purple/"
    virus_interpreter_directory = "virusinterpreter/"

    _data: dict[str, dict[str, str]] = {}

    def __init__(
        self,
        prefix: str,
        output_dir: str,
        bucket: str = "org.umccr.data.oncoanalyser",
        **kwargs,
    ) -> None:
        """
        Initialize this class.

        :param prefix: the bucket prefix.
        :param bucket: the bucket to download from.
        :param output_dir: the directory to save downloaded files to.
        """
        self.__dict__.update(kwargs)

        self._output_dir = output_dir
        self._downloader = Downloader(bucket, prefix, "s3")

    def prepare(self) -> None:
        """
        Prepares all sample data by downloading it from the s3 bucket.
        """
        self._data["chord"] = self._downloader.sync(
            os.path.join(self._output_dir, self.chord_directory),
            self.chord_directory,
            False,
        )
        self._data["lilac"] = self._downloader.sync(
            os.path.join(self._output_dir, self.lilac_directory),
            self.lilac_directory,
            False,
        )
        self._data["purple"] = self._downloader.sync(
            os.path.join(self._output_dir, self.purple_directory),
            self.purple_directory,
            False,
        )
        self._data["virusinterpreter"] = self._downloader.sync(
            os.path.join(self._output_dir, self.virus_interpreter_directory),
            self.virus_interpreter_directory,
            False,
        )
