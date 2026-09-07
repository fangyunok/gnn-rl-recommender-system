from __future__ import annotations

import argparse
import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "ml-1m.zip"
    if not archive.exists():
        context = ssl.create_default_context()
        with urllib.request.urlopen(URL, context=context) as response, archive.open("wb") as target:
            shutil.copyfileobj(response, target)
    with zipfile.ZipFile(archive) as source:
        source.extractall(output_dir)
    return output_dir / "ml-1m"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    print(download(args.output_dir))

