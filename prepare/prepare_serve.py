from prepare.downloader import Downloader


class PrepareServe:
    """
    Prepare all data related to SERVE.
    """

    _characteristics_name = "ActionableCharacteristics.38.tsv"
    _fusions_name = "ActionableFusions.38.tsv"
    _genes_name = "ActionableGenes.38.tsv"
    _hla_name = "ActionableHLA.38.tsv"
    _hotspots_name = "ActionableHotspots.38.tsv"
    _ranges_name = "ActionableRanges.38.tsv"

    _data: dict[str, str] = {}

    def __init__(
        self,
        url: str = "https://storage.googleapis.com/hmf-public/HMFtools-Resources/serve/38/",
        output_dir: str = "data/serve/",
    ) -> None:
        """
        Initialize this class.

        :param url: the url to download data from.
        :param output_dir: the directory to save downloaded files to.
        """
        self._output_dir = output_dir
        self._downloader = Downloader(url)

    @property
    def characteristics(self) -> str:
        """
        Get the characteristics tsv.

        :return: characteristics tsv.
        """
        return self._data["characteristics"]

    @property
    def fusions(self) -> str:
        """
        Get the fusions tsv.

        :return: characteristics tsv.
        """
        return self._data["fusions"]

    @property
    def genes(self) -> str:
        """
        Get the genes tsv.

        :return: genes tsv.
        """
        return self._data["genes"]

    @property
    def hla(self) -> str:
        """
        Get the HLA tsv.

        :return: HLA tsv.
        """
        return self._data["hla"]

    @property
    def hotspots(self) -> str:
        """
        Get the hotspots tsv.

        :return: hotspots tsv.
        """
        return self._data["hotspots"]

    @property
    def ranges(self) -> str:
        """
        Get the ranges tsv.

        :return: ranges tsv.
        """
        return self._data["ranges"]

    def prepare(self) -> None:
        """
        Prepares all SERVE data by downloading it from the hmftools google storage or reading it
        from a file if present. Writes downloaded data to a file.
        """

        self._data["characteristics"] = self._downloader.get_or_download(
            self._output_dir, self._characteristics_name
        )
        self._data["fusions"] = self._downloader.get_or_download(
            self._output_dir, self._fusions_name
        )
        self._data["genes"] = self._downloader.get_or_download(
            self._output_dir, self._genes_name
        )
        self._data["hla"] = self._downloader.get_or_download(
            self._output_dir, self._hla_name
        )
        self._data["hotspots"] = self._downloader.get_or_download(
            self._output_dir, self._hotspots_name
        )
        self._data["ranges"] = self._downloader.get_or_download(
            self._output_dir, self._ranges_name
        )
