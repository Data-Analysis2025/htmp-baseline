"""Download and place the calendar data used for cyclic date features."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import kagglehub

DEFAULT_DATASET = "ambrosm/s-and-p-historical-data-for-hull-tactical-competition"
DEFAULT_OUTPUT = Path("data/calendar_data.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Kaggle dataset slug to download.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination path for the calendar CSV.",
    )
    parser.add_argument(
        "--calendar-file",
        type=str,
        default=None,
        help="Explicit calendar CSV filename inside the dataset. If omitted, the first '*calendar*.csv' is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_dir = Path(kagglehub.dataset_download(args.dataset))
    print(f"Downloaded dataset to: {download_dir}")

    if args.calendar_file:
        calendar_source = download_dir / args.calendar_file
        if not calendar_source.exists():
            raise FileNotFoundError(f"Specified calendar file not found: {calendar_source}")
    else:
        candidates = sorted(download_dir.glob("**/*historical*.csv"))
        if not candidates:
            raise FileNotFoundError("Calendar CSV not found in the downloaded dataset.")
        calendar_source = candidates[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(calendar_source, args.output)
    print(f"Copied calendar data to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
