"""Data loading utilities."""

from pathlib import Path
import tarfile
import urllib.request
import pandas as pd


def load_csv_from_tar(url: str, tar_file: str, data_dir: Path, csv_path: str) -> pd.DataFrame:
    """
    Download tar/tgz file, extract it, and load CSV.

    Args:
        url: URL to download from
        tar_file: Tar filename (e.g., "housing.tgz")
        data_dir: Directory to save/extract files
        csv_path: Path to CSV within extracted archive (e.g., "housing/housing.csv")

    Returns:
        Loaded DataFrame
    """
    tarball_path = data_dir / tar_file
    if not tarball_path.is_file():
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as tarball:
            tarball.extractall(data_dir, filter="data")
    return pd.read_csv(data_dir / csv_path)
