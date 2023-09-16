import os

from classifer.command import Command
from pathlib import Path


class Protect:
    """
    Run linx on a sample.
    """

    chord_directory = "chord/"
    lilac_directory = "lilac/"
    purple_directory = "purple/"
    virus_interpreter_directory = "virusinterpreter/"

    annotated_virus_tsv_ending = ".virus.annotated.tsv"
    chord_prediction_txt_ending = "_chord_prediction.txt"

    lilac_qc_csv_ending = ".lilac.qc.csv"
    lilac_result_csv_ending = ".lilac.csv"

    linx_breakend_tsv_ending = ".linx.breakend.tsv"
    linx_driver_catalog_tsv_ending = ".linx.driver.catalog.tsv"
    linx_fusion_tsv_ending = ".linx.fusion.tsv"

    purple_gene_copy_number_tsv_ending = ".purple.cnv.gene.tsv"
    purple_germline_driver_catalog_tsv_ending = ".driver.catalog.somatic.tsv"
    purple_germline_variant_vcf_ending = ".purple.germline.vcf.gz"
    purple_purity_tsv_ending = ".purple.purity.tsv"
    purple_qc_file_ending = ".purple.qc"
    purple_somatic_driver_catalog_tsv_ending = ".driver.catalog.somatic.tsv"
    purple_somatic_variant_vcf_ending = ".purple.somatic.vcf.gz"

    def __init__(
        self,
        tumor_sample_id: str,
        reference_sample_id: str,
        sample_dir: str,
        linx_dir: str,
        purple_dir: str,
        output_dir: str,
        jar_file: str = "data/software/linx-v1.21/linx_v1.21.jar",
        doid_json: str = "data/reference/disease_ontology/doid.json",
        serve_dir: str = "data/serve/",
        driver_gene_panel: str = "data/reference/DriverGenePanel.38.tsv",
        **kwargs
    ) -> None:
        """
        Initialize this class.

        :param tumor_sample_id: tumor sample id
        :param reference_sample_id: reference sample id
        :param sample_dir: sample directory
        :param linx_dir: linx directory
        :param purple_dir: purple directory
        :param output_dir: output directory
        :param jar_file: linx jar file
        :param doid_json: doid json
        :param serve_dir: serve directory
        :param driver_gene_panel: driver gene panel
        """
        self.__dict__.update(kwargs)

        self._tumor_sample_id = tumor_sample_id
        self._reference_sample_id = reference_sample_id
        self._sample_dir = sample_dir
        self._linx_dir = linx_dir
        self._purple_dir = purple_dir

        self._output_dir = output_dir
        self._jar_file = jar_file
        self._doid_json = doid_json
        self._serve_dir = serve_dir
        self._driver_gene_panel = driver_gene_panel

        self._annotated_virus_tsv = self.find_file(
            self.virus_interpreter_directory,
            "*" + self.annotated_virus_tsv_ending,
        )
        self._chord_prediction_txt = self.find_file(
            self.chord_directory,
            "*" + self.chord_prediction_txt_ending,
        )

        self._lilac_qc_csv = self.find_file(
            self.lilac_directory,
            "*" + self.lilac_qc_csv_ending,
        )
        self._lilac_result_csv = self.find_file(
            self.lilac_directory,
            "*" + self.lilac_result_csv_ending,
        )

        self._linx_breakend_tsv = self.find_file(
            self._linx_dir,
            "*" + self.linx_breakend_tsv_ending,
        )
        self.linx_driver_catalog_tsv = self.find_file(
            self._linx_dir,
            "*" + self.linx_driver_catalog_tsv_ending,
        )
        self._linx_fusion_tsv = self.find_file(
            self._linx_dir,
            "*" + self.linx_fusion_tsv_ending,
        )

        self._purple_gene_copy_number_tsv = self.find_file(
            self._purple_dir,
            "*" + self.purple_gene_copy_number_tsv_ending,
        )
        self._purple_germline_driver_catalog_tsv = self.find_file(
            self._purple_dir,
            "*" + self.purple_germline_driver_catalog_tsv_ending,
        )
        self._purple_germline_variant_vcf = self.find_file(
            self._purple_dir,
            "*" + self.purple_germline_variant_vcf_ending,
        )
        self._purple_purity_tsv = self.find_file(
            self._purple_dir,
            "*" + self.purple_purity_tsv_ending,
        )
        self._purple_qc_file = self.find_file(
            self._purple_dir,
            "*" + self.purple_qc_file_ending,
        )
        self._purple_somatic_driver_catalog_tsv = self.find_file(
            self._purple_dir,
            "*" + self.purple_somatic_driver_catalog_tsv_ending,
        )
        self._purple_somatic_variant_vcf = self.find_file(
            self._purple_dir,
            "*" + self.purple_somatic_variant_vcf_ending,
        )

    def find_file(self, in_dir: str, file: str) -> str:
        """
        Find a file in a directory.

        :param in_dir: directory to search
        :param file: file to search for
        :return: path to file
        """
        for path in Path(os.path.join(self._sample_dir, in_dir)).rglob(file):
            return str(path)

    def run(self) -> None:
        """
        Run linx
        """

        command = Command(self._output_dir)

        command.add_arg("java")
        command.add_arg("-jar")
        command.add_arg(self._jar_file)
        command.add_arg("-tumor_sample_id")
        command.add_arg(self._tumor_sample_id)
        command.add_arg("-reference_sample_id")
        command.add_arg(self._reference_sample_id)

        command.add_arg("-primary_tumor_doids")
        command.add_arg("''")
        command.add_arg("-ref_genome_version")
        command.add_arg("38")

        command.add_arg("-annotated_virus_tsv")
        command.add_arg(self._annotated_virus_tsv)
        command.add_arg("-chord_prediction_txt")
        command.add_arg(self._chord_prediction_txt)

        command.add_arg("-lilac_qc_csv")
        command.add_arg(self._lilac_qc_csv)
        command.add_arg("-lilac_result_csv")
        command.add_arg(self._lilac_result_csv)

        command.add_arg("-linx_breakend_tsv")
        command.add_arg(self._linx_breakend_tsv)
        command.add_arg("-linx_driver_catalog_tsv")
        command.add_arg(self.linx_driver_catalog_tsv)
        command.add_arg("-linx_fusion_tsv")
        command.add_arg(self._linx_fusion_tsv)

        command.add_arg("-purple_gene_copy_number_tsv")
        command.add_arg(self._purple_gene_copy_number_tsv)
        command.add_arg("-purple_germline_driver_catalog_tsv")
        command.add_arg(self._purple_germline_driver_catalog_tsv)
        command.add_arg("-purple_germline_variant_vcf")
        command.add_arg(self._purple_germline_variant_vcf)
        command.add_arg("-purple_purity_tsv")
        command.add_arg(self._purple_purity_tsv)
        command.add_arg("-purple_qc_file")
        command.add_arg(self._purple_qc_file)
        command.add_arg("-purple_somatic_driver_catalog_tsv")
        command.add_arg(self._purple_somatic_driver_catalog_tsv)
        command.add_arg("-purple_somatic_variant_vcf")
        command.add_arg(self._purple_somatic_variant_vcf)

        command.add_arg("-driver_gene_tsv")
        command.add_arg(self._driver_gene_panel)
        command.add_arg("-doid_json")
        command.add_arg(self._doid_json)
        command.add_arg("-serve_actionability_dir")
        command.add_arg(self._serve_dir)

        command.add_arg("-log_debug")
        command.add_arg("-output_dir")
        command.add_arg(self._output_dir)

        command.run()
