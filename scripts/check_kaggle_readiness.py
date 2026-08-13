"""Print a Kaggle readiness report for the ARC harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arc_harness.kaggle import check_kaggle_readiness


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", default="arc_harness")
    parser.add_argument("--environment-files")
    parser.add_argument("--model-config")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any check fails.")
    args = parser.parse_args()

    report = check_kaggle_readiness(
        package_root=args.package_root,
        environments_dir=args.environment_files,
        model_config=args.model_config,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.ok or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())
