"""Build a Kaggle-ready copy of the ARC harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arc_harness.integrations.kaggle import build_kaggle_package


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--package-root", default="arc_harness")
    parser.add_argument("--no-scripts", action="store_true")
    args = parser.parse_args()
    package = build_kaggle_package(
        args.output_dir,
        package_root=args.package_root,
        include_scripts=not args.no_scripts,
    )
    print(json.dumps(package.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
