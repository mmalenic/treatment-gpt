from pathlib import Path


def find_file(in_dir: str, file: str) -> str:
    """
    Find a file in a directory.

    :param in_dir: directory to search
    :param file: file to search for
    :return: path to file
    """
    for path in Path(in_dir).rglob(file):
        return str(path)
