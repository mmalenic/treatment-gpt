import urllib.request
from pathlib import Path
from typing import Literal, Optional


class Downloader:
    """
    Downloads data from a url or bucket.
    """

    def __init__(
        self, url: str, prefix: str = "", mode: Literal["url", "s3"] = "url"
    ) -> None:
        """
        Initialize this class.

        :param url: the url or bucket to download from.
        :param prefix: the prefix to add to the url or bucket.
        :param mode: the mode of operation
        """
        self.url = url
        self.prefix = prefix
        self.mode = mode

    def download(self, name: str, save_to: Optional[str | Path] = None) -> str | None:
        """
        Download data from the url, prefix and data name.

        :param name: the name of the file to download
        :param save_to: save to the file instead of reading to a string.

        :return: return the downloaded data or None if saving.
        """
        if self.mode == "url":
            if save_to is None:
                with urllib.request.urlopen(self.url + self.prefix + name) as f:
                    return f.read()
            else:
                urllib.request.urlretrieve(self.url + self.prefix + name, save_to)
        else:
            raise NotImplementedError

    def get_or_download(
        self, output_dir: str, name: str, read_data: bool = True
    ) -> str | None:
        """
        Get the data from the output_dir or download and save it to the output directory.

        :param output_dir: the directory to save and get data from.
        :param name: the name of the data.
        :param read_data: whether to read the data or just save it.

        :return: the data or None if just saved.
        """
        output_file = Path(output_dir + name)

        if not output_file.exists():
            output_file.parent.mkdir(exist_ok=True, parents=True)

            if read_data:
                output = self.download(name)
                output_file.write_text(output)
                return output
            else:
                self.download(name, output_file)
        elif read_data:
            return output_file.read_text()
