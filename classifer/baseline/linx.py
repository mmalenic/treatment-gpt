from classifer.command import Command


class RunLinx:
    """
    Run linx on a sample.
    """

    def __init__(
        self,
        sample_id: str,
        purple_directory: str,
        purple_sv_vcf_file: str,
        output_dir: str,
        jar_file: str = "data/software/linx-v1.21/linx_v1.21.jar",
        ensembl_data_directory: str = "data/reference/ensembl",
        known_fusion_file: str = "data/reference/known_fusion_data.38.csv",
        driver_gene_panel: str = "data/other/DriverGenePanel.38.tsv",
    ) -> None:
        """
        Initialize this class.

        :param sample_id: sample id
        :param purple_directory: purple directory
        :param purple_sv_vcf_file: purple sv vcf file
        :param output_dir: output directory
        :param jar_file: linx jar file
        :param ensembl_data_directory: ensembl data
        :param known_fusion_file: fusion file
        :param driver_gene_panel: driver gene panel
        """
        self._sample_id = sample_id
        self._purple_directory = purple_directory
        self._purple_sv_vcf_file = purple_sv_vcf_file
        self._output_dir = output_dir
        self._jar_file = jar_file
        self._ensembl_data_directory = ensembl_data_directory
        self._known_fusion_file = known_fusion_file
        self._driver_gene_panel = driver_gene_panel

    def run(self) -> None:
        """
        Run linx
        """

        command = Command(self._output_dir)

        command.add_arg("java")
        command.add_arg("-jar")
        command.add_arg(self._jar_file)
        command.add_arg("-sample")
        command.add_arg(self._sample_id)
        command.add_arg("-sv_vcf")
        command.add_arg(self._purple_sv_vcf_file)
        command.add_arg("-purple_dir")
        command.add_arg(self._purple_directory)
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
