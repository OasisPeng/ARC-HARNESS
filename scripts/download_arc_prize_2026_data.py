"""Download ARC Prize 2026 public Kaggle data."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


COMPETITION = "arc-prize-2026-arc-agi-3"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/arc-prize-2026-arc-agi-3")
    parser.add_argument("--kaggle-bin", default=shutil.which("kaggle") or _user_kaggle_bin())
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"{COMPETITION}.zip"
    if archive.exists() and not args.force:
        print(f"Using existing archive: {archive}")
    else:
        command = [args.kaggle_bin, "competitions", "download", "-c", COMPETITION, "-p", str(output)]
        if args.force:
            command.append("--force")
        _run(command)

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(output)

    metadata_count = len(list((output / "environment_files").rglob("metadata.json")))
    wheel_count = len(list((output / "arc_agi_3_wheels").glob("*.whl")))
    print(f"Downloaded data to {output}")
    print(f"public_games={metadata_count} wheels={wheel_count}")
    return 0


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _user_kaggle_bin() -> str:
    return str(Path.home() / "Library" / "Python" / f"{sys.version_info.major}.{sys.version_info.minor}" / "bin" / "kaggle")


if __name__ == "__main__":
    raise SystemExit(main())
