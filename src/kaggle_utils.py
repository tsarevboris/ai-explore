import zipfile
import subprocess

from pathlib import Path
from typing import List

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _unzip(zip_path: Path, extract_to: Path) -> List[Path]:
    _ensure_dir(extract_to)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    return [extract_to / name for name in zf.namelist()]

def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

def download_kaggle_competition(
    comp: str,
    raw_dir: str | Path = "data/raw",
    out_subdir: str | None = None,
    force: bool = False,
) -> List[Path]:
    """
    Download a Kaggle *competition* bundle (zip) and unzip it.
    Returns list of extracted file Paths.

    comp: competition slug, e.g. "titanic"
    raw_dir: base directory for raw data
    out_subdir: optional subfolder under raw_dir; default = comp name
    force: if True, re-download even if zip already exists
    """
    
    raw_dir = Path(raw_dir)
    _ensure_dir(raw_dir)

    out_subdir = out_subdir or comp
    extract_dir = raw_dir / out_subdir
    zip_path = raw_dir / f"{comp}.zip"

    # Download zip if needed
    if force or not zip_path.exists():
        # kaggle competitions download -c titanic -p data/raw
        _run(["kaggle", "competitions", "download", "-c", comp, "-p", str(raw_dir)])

    # Unzip if needed
    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        return _unzip(zip_path, extract_dir)
    else:
        return list(extract_dir.iterdir())

def download_kaggle_dataset(
    dataset: str,
    raw_dir: str | Path = "data/raw",
    out_subdir: str | None = None,
    force: bool = False,
) -> List[Path]:
    """
    Download a Kaggle *dataset* (not competition) and unzip it.
    Returns list of extracted file Paths.

    dataset: e.g. "camnugent/california-housing-prices"
    raw_dir: base directory for raw data
    out_subdir: optional subfolder under raw_dir; default = last part of dataset slug
    force: if True, re-download even if zip already exists
    """
    raw_dir = Path(raw_dir)
    _ensure_dir(raw_dir)

    # use the last slug token as default folder name
    default_name = dataset.split("/")[-1]
    out_subdir = out_subdir or default_name
    extract_dir = raw_dir / out_subdir
    zip_path = raw_dir / f"{default_name}.zip"

    # Download zip if needed
    if force or not zip_path.exists():
        # example: kaggle datasets download -d camnugent/california-housing-prices -p data/raw
        _run(["kaggle", "datasets", "download", "-d", dataset, "-p", str(raw_dir)])

    # Unzip if needed
    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        return _unzip(zip_path, extract_dir)
    else:
        return list(extract_dir.iterdir())
    