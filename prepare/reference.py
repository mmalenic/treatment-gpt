import os

from prepare.downloader import Downloader


class Reference:
    """
    Prepare reference data such as disease ontology and ensembl data.
    """

    doid_name = "disease_ontology/doid.json"
    driver_gene_panel_name = "dna_pipeline/common/DriverGenePanel.38.tsv"
    known_fusion_data_name = "dna_pipeline/sv/known_fusion_data.38.csv"
    ensembl_directory_name = "dna_pipeline/common/ensembl_data/"

    def __init__(
        self,
        bucket: str = "umccr-refdata-dev",
        prefix: str = "workflow_data/hmf_reference_data/hmftools/5.33_38--0/",
        output_dir: str = "data/reference/",
        ensembl_data_directory: str = "ensembl",
        **kwargs
    ) -> None:
        """
        Initialize this class.

        :param bucket: the bucket to download from.
        :param prefix: the bucket prefix.
        :param output_dir: the directory to save downloaded files to.
        """
        self.__dict__.update(kwargs)

        self._output_dir = output_dir
        self._downloader = Downloader(bucket, prefix, "s3")
        self._ensembl_data_directory = ensembl_data_directory

        self._data = {}

    @property
    def doid(self) -> str:
        """
        Get the disease ontology json.

        :return: disease ontology json.
        """
        return self._data["doid"]

    @property
    def driver_gene_panel(self) -> str:
        """
        Get the driver gene panel tsv.

        :return: driver gene panel tsv.
        """
        return self._data["driver_gene_panel"]

    @property
    def known_fusion_data(self) -> str:
        """
        Get the known fusion data csv.

        :return: known fusion data csv.
        """
        return self._data["known_fusion_data"]

    @property
    def ensembl_data(self) -> dict[str, str]:
        """
        Get the ensemble data.

        :return: ensemble data.
        """
        return self._data["ensembl_data"]

    def prepare(self) -> None:
        """
        Prepares all other data by downloading it from the s3 bucket.
        """

        self._data["doid"] = self._downloader.get(self._output_dir, self.doid_name)
        self._data["driver_gene_panel"] = self._downloader.get(
            self._output_dir, self.driver_gene_panel_name
        )
        self._data["known_fusion_data"] = self._downloader.get(
            self._output_dir, self.known_fusion_data_name
        )

        self._data["ensembl_data"] = self._downloader.sync(
            os.path.join(self._output_dir, self._ensembl_data_directory),
            [self.ensembl_directory_name],
            self.ensembl_directory_name,
        )
