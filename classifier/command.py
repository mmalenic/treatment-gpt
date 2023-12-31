import os
import subprocess
from pathlib import Path


class Command:
    """
    Run a command.
    """

    def __init__(
        self,
        output_dir: str,
    ) -> None:
        """
        Initialize this class.

        :param output_dir: output directory
        """
        self._output_dir = output_dir
        self._commands = []

    def add_arg(self, command: str) -> None:
        """
        Add a command to run.
        :param command: command
        """

        self._commands += [command]

    def run(self) -> None:
        """
        Run the commands.
        """

        output_dir = Path(self._output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        with open(os.path.join(output_dir, "log.txt"), "w") as f:
            subprocess.run(self._commands, stdout=f, stderr=subprocess.STDOUT)
