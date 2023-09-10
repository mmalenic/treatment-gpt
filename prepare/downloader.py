import urllib.request
from pathlib import Path
from typing import Literal


class Downloader:
    """
    Downloads data from a url or bucket.
    """

    def __init__(self, url: str, prefix: str = "", mode: Literal["url", "s3"] = "url") -> None:
        """
        Initialize this class.

        :param url: the url or bucket to download from.
        :param prefix: the prefix to add to the url or bucket.
        :param mode: the mode of operation
        """
        self.url = url
        self.prefix = prefix
        self.mode = mode

    def download(self, name: str) -> str:
        """
        Download data from the url, prefix and data name.

        :return: return the downloaded data.
        """

        if self.mode == "url":
            return urllib.request.urlopen(self.url + self.prefix + name).read()
        else:
            raise NotImplementedError

    def get_or_download(self, output_dir: str, name: str) -> str:
        """
        Get the data from the output_dir or download and save it to the output directory.

        :param output_dir: the directory to save and get data from.
        :param name: the name of the data.
        :return: the data
        """
        output_file = Path(output_dir + name)

        if not output_file.exists():
            output_file.parent.mkdir(exist_ok=True, parents=True)

            output = self.download(name)
            output_file.write_text(output)

            return output
        else:
            return output_file.read_text()