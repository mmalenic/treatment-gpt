import urllib.request
from pathlib import Path


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

    def __init__(self, url: str, output_dir: str = "data/serve/") -> None:
        """
        Initialize this class.

        :param url: the url to download data from.
        :param output_dir: the directory to save downloaded files to.
        """
        self.url = url
        self.output_dir = output_dir

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
        self._fetch_and_write("characteristics", self._characteristics_name)
        self._fetch_and_write("fusions", self._fusions_name)
        self._fetch_and_write("genes", self._genes_name)
        self._fetch_and_write("hla", self._hla_name)
        self._fetch_and_write("hotspots", self._hotspots_name)
        self._fetch_and_write("ranges", self._hotspots_name)

    def _fetch_and_write(self, write_to: str, name: str) -> None:
        output_file = Path(self.output_dir + name)

        if not output_file.exists():
            output_file.parent.mkdir(exist_ok=True, parents=True)

            output = urllib.request.urlopen(self.url + name).read()
            output_file.write_text(output)

            self._data[write_to] = output
        else:
            output = output_file.read_text()

            self._data[write_to] = output
