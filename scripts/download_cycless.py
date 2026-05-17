"""
Download and extract the CYCleSS real UK crop yield dataset.
CC0-1.0 license — free for any use.
"""

import zipfile
from pathlib import Path

import requests

FIGSHARE_URL = "https://ndownloader.figshare.com/files/49785804"
ZIP_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "cycless.zip"
EXTRACT_DIR = ZIP_PATH.parent / "cycless"


def download(force: bool = False) -> Path:
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists() and not force:
        print(f"Already downloaded: {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1024:.0f} KB)")
        return ZIP_PATH

    print(f"Downloading CYCleSS dataset from {FIGSHARE_URL} ...")
    resp = requests.get(FIGSHARE_URL, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(ZIP_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {pct:.0f}% ({downloaded / 1024:.0f} KB)", end="")
    print()
    print(f"Saved to {ZIP_PATH}")
    return ZIP_PATH


def extract() -> Path:
    if EXTRACT_DIR.exists():
        print(f"Already extracted: {EXTRACT_DIR}")
        return EXTRACT_DIR

    print(f"Extracting to {EXTRACT_DIR} ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(EXTRACT_DIR)
    print(f"Extracted {len(list(EXTRACT_DIR.rglob('*')))} files")
    return EXTRACT_DIR


if __name__ == "__main__":
    download()
    extract()
    print("Done. Run `python training/prepare_real_data.py` to process.")
