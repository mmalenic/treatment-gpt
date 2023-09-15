from prepare.downloader import Downloader


class Sample:
    """
    Prepare subject data for running PROTECT
    """

    _chord_directory = "chord/"
    _lilac_directory = "lilac/"
    _purple_directory = "purple/"
    _virus_interpreter_directory = "virusinterpreter/"

    _data: dict[str, dict[str, str]] = {}

    def __init__(
        self,
        prefix: str,
        output_dir: str,
        bucket: str = "org.umccr.data.oncoanalyser",
    ) -> None:
        """
        Initialize this class.

        :param prefix: the bucket prefix.
        :param bucket: the bucket to download from.
        :param output_dir: the directory to save downloaded files to.
        """
        self._output_dir = output_dir
        self._downloader = Downloader(bucket, prefix, "s3")

    def prepare(self) -> None:
        """
        Prepares all sample data by downloading it from the s3 bucket.
        """
        self._data["chord"] = self._downloader.sync(
            self._output_dir + self._chord_directory, self._chord_directory
        )
        self._data["lilac"] = self._downloader.sync(
            self._output_dir + self._lilac_directory, self._lilac_directory
        )
        self._data["purple"] = self._downloader.sync(
            self._output_dir + self._purple_directory, self._purple_directory
        )
        self._data["virusinterpreter"] = self._downloader.sync(
            self._output_dir + self._virus_interpreter_directory,
            self._virus_interpreter_directory,
        )
