import os
import urllib.request
from pathlib import Path
from typing import Literal, Optional, List
import boto3


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
                url = self.url + self.prefix + name
                print("downloading url file:", url)

                with urllib.request.urlopen(url) as f:
                    return f.read().decode("utf-8")
            else:
                print("downloading url file:", save_to)
                urllib.request.urlretrieve(self.url + self.prefix + name, save_to)
        else:
            s3 = boto3.resource("s3")

            if save_to is None:
                obj = s3.Object(self.url, self.prefix + name)

                print("downloading object file:", obj)
                return obj.get()["Body"].read().decode("utf-8")
            else:
                bucket = s3.Bucket(self.url)

                print("downloading object file:", save_to)
                bucket.download_file(self.prefix + name, save_to)

    def sync(
        self,
        output_dir: str,
        additional_prefixes: List[str],
        find_on_prefix: Optional[str],
        read_data: bool = True,
        lazy_check: bool = False,
    ) -> dict[str, str] | None:
        """
        Sync the data from the prefix in S3 to the output_dir.
        This function returns None is the mode is not s3.

        :param find_on_prefix: find the file for this prefix.
        :param output_dir: the directory to save and get data from.
        :param additional_prefixes: an additional prefixes for filtering in the s3 bucket.
        :param read_data: whether to read the data or just save it.
        :param lazy_check: check to see if the folder exists rather than listing all objects in s3.

        :return: the data or None if just saved.
        """
        if self.mode != "s3":
            return None

        output_dir = Path(output_dir)

        if lazy_check and output_dir.exists():
            return None

        output_dir.mkdir(exist_ok=True, parents=True)

        s3 = boto3.resource("s3")
        bucket = s3.Bucket(self.url)

        objects = bucket.objects.filter(Prefix=self.prefix)
        objects = [
            obj
            for obj in objects
            if all([(prefix in obj.key) for prefix in additional_prefixes])
            and not obj.key.endswith("/")
        ]
        print("objects:", objects)

        output = {}
        for obj in objects:
            if find_on_prefix is not None:
                file = obj.key[obj.key.find(find_on_prefix) + len(find_on_prefix) :]
            else:
                file = obj.key[obj.key.rfind("/") + 1 :]

            if file is None or file == "":
                continue

            local_file = Path(os.path.join(output_dir, file))
            local_file.parent.mkdir(exist_ok=True, parents=True)

            if not local_file.exists():
                print("downloading object file:", local_file)
                bucket.download_file(obj.key, local_file)

            if read_data:
                with open(local_file, "r", encoding="utf-8") as f:
                    output[file] = f.read()

        if read_data:
            return output

    def get(self, output_dir: str, name: str, read_data: bool = True) -> str | None:
        """
        Get the data from the output_dir or download and save it to the output directory.

        :param output_dir: the directory to save and get data from.
        :param name: the name of the data.
        :param read_data: whether to read the data or just save it.

        :return: the data or None if just saved.
        """
        output_file = Path(os.path.join(output_dir, name))

        if not output_file.exists():
            output_file.parent.mkdir(exist_ok=True, parents=True)

            if read_data:
                output = self.download(name)
                output_file.write_text(output, encoding="utf-8")
                return output
            else:
                self.download(name, output_file)
        elif read_data:
            return output_file.read_text()
