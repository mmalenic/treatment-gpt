from pathlib import Path

from classifier.command import Command
from classifier.util import find_file, check_prerequisite
import os
import pandas as pd


class Linx:
    """
    Run linx on a sample.
    """

    purple_sv_file_ending = ".purple.sv.vcf.gz"
    purple_directory = "purple/"
    linx_directory = "linx/"

    def __init__(
        self,
        sample_dir: str,
        jar_file: str = "data/software/linx-v1.21/linx_v1.21.jar",
        ensembl_data_directory: str = "data/reference/ensembl",
        known_fusion_file: str = "data/reference/dna_pipeline/sv/known_fusion_data.38.csv",
        driver_gene_panel: str = "data/reference/dna_pipeline/common/DriverGenePanel.38.tsv",
        **kwargs
    ) -> None:
        """
        Initialize this class.

        :param sample_dir: sample directory
        :param jar_file: linx jar file
        :param ensembl_data_directory: ensembl data
        :param known_fusion_file: fusion file
        :param driver_gene_panel: driver gene panel
        """
        self.__dict__.update(kwargs)

        self._sample_dir = sample_dir

        sample_sheet = pd.read_csv(os.path.join(sample_dir, "samplesheet.csv"))
        self._sample_id = sample_sheet[
            (sample_sheet["sample_type"] == "tumor")
            & (sample_sheet["sequence_type"] == "wgs")
        ]["sample_name"].item()

        self._sample_id = os.path.basename(os.path.normpath(self._sample_dir))
        self._sample_dir = sample_dir
        self._purple_sv_vcf_file = find_file(
            os.path.join(self._sample_dir, self.purple_directory),
            "*" + self.purple_sv_file_ending,
        )
        self._output_dir = os.path.join(sample_dir, self.linx_directory)
        Path(self._output_dir).mkdir(parents=True, exist_ok=True)

        self._jar_file = jar_file
        self._ensembl_data_directory = ensembl_data_directory
        self._known_fusion_file = known_fusion_file
        self._driver_gene_panel = driver_gene_panel

    def run(self) -> None:
        """
        Run linx
        """
        print("running linx for:", self._sample_id)

        command = Command(self._output_dir)

        command.add_arg("java")
        command.add_arg("-jar")
        command.add_arg(self._jar_file)
        command.add_arg("-sample")
        command.add_arg(self._sample_id)
        check_prerequisite(self._purple_sv_vcf_file, "sv_vcf")
        command.add_arg("-sv_vcf")
        command.add_arg(self._purple_sv_vcf_file)
        command.add_arg("-purple_dir")
        command.add_arg(os.path.join(self._sample_dir, self.purple_directory))
        command.add_arg("-ref_genome_version")
        command.add_arg("38")
        command.add_arg("-ensembl_data_dir")
        command.add_arg(self._ensembl_data_directory)
        command.add_arg("-check_fusions")
        command.add_arg("-known_fusion_file")
        command.add_arg(self._known_fusion_file)
        command.add_arg("-check_drivers")
        command.add_arg("-driver_gene_panel")
        command.add_arg(self._driver_gene_panel)
        command.add_arg("-write_vis_data")
        command.add_arg("-log_debug")
        command.add_arg("-output_dir")
        command.add_arg(self._output_dir)

        command.run()
