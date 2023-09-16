from prepare.downloader import Downloader


class Software:
    """
    Prepare all software
    """

    protect_jar = "protect-v2.3/protect.jar"
    linx_jar = "linx-v1.21/linx_v1.21.jar"

    def __init__(
        self,
        url: str = "https://github.com/hartwigmedical/hmftools/releases/download/",
        output_dir: str = "data/software/",
        **kwargs
    ) -> None:
        """
        Initialize this class.

        :param url: the url to download data from.
        :param output_dir: the directory to save downloaded files to.
        """
        self.__dict__.update(kwargs)

        self._output_dir = output_dir
        self._downloader = Downloader(url)

    def prepare(self) -> None:
        """
        Prepares all software data by downloading it from the hmftools GitHub or checking
        that it exists in the output_dir.
        """
        self._downloader.get(self._output_dir, self.protect_jar, False)
        self._downloader.get(self._output_dir, self.linx_jar, False)
