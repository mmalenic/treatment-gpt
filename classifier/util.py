from pathlib import Path


def find_file(in_dir: str, file: str) -> str | None:
    """
    Find a file in a directory.

    :param in_dir: directory to search
    :param file: file to search for
    :return: path to file
    """
    for path in Path(in_dir).rglob(file):
        return str(path)

    return None


def check_prerequisite(path: str, message: str) -> None:
    """
    Check if the path exists or raise exception.
    :param path: path to check
    :param message: message
    """
    if path is None or not Path(path).exists():
        raise Exception("{0} does not exist: {1}".format(message, path))


def accuracy_score(y_true, y_pred) -> float:
    """
    Calculate accuracy score, i.e. hamming score.
    """
    try:
        return ((y_true & y_pred).sum(axis=1) / (y_true | y_pred).sum(axis=1)).mean()
    except ZeroDivisionError as e:
        print("error from accuracy score:", e)
        return 0
