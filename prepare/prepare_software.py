from prepare.downloader import Downloader


class PrepareSoftware:
    """
    Prepare all software
    """

    _protect_jar = "protect-v2.3/protect.jar"
    _linx_jar = "linx-v1.21/linx_v1.21.jar"

    def __init__(
        self,
        url: str = "https://github.com/hartwigmedical/hmftools/releases/download/",
        output_dir: str = "data/software/",
    ) -> None:
        """
        Initialize this class.

        :param url: the url to download data from.
        :param output_dir: the directory to save downloaded files to.
        """
        self._output_dir = output_dir
        self._downloader = Downloader(url)

    def prepare(self) -> None:
        """
        Prepares all software data by downloading it from the hmftools GitHub or checking
        that it exists in the output_dir.
        """
        self._downloader.get_or_download(self._output_dir, self._protect_jar, False)
        self._downloader.get_or_download(self._output_dir, self._linx_jar, False)
