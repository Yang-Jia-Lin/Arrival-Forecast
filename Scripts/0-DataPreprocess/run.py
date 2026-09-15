"""Run one or all dataset preprocessors, then validate their outputs."""

import argparse

import alibaba_2020
import azure_2019
import google_2011
import google_2019
import philly_2017
from validate import validate


PROCESSORS = {
    "alibaba_2020": alibaba_2020.main,
    "philly_2017": philly_2017.main,
    "google_2011": google_2011.main,
    "google_2019": google_2019.main,
    "azure_2019": azure_2019.main,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess task-arrival datasets from local raw files."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        metavar="DATASET",
        help="dataset names; omit to process all datasets",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="regenerate the overview figure after preprocessing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = args.datasets or list(PROCESSORS)
    unknown = sorted(set(selected) - PROCESSORS.keys())
    if unknown:
        choices = ", ".join(PROCESSORS)
        raise SystemExit(f"Unknown dataset(s): {', '.join(unknown)}. Choose: {choices}")

    for name in selected:
        print(f"\n[{name}] preprocessing")
        PROCESSORS[name]()
    print("\n[validation]")
    validate(selected)
    if args.plot:
        import plot

        print("\n[plot]")
        plot.main()


if __name__ == "__main__":
    main()
