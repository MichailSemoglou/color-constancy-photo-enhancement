"""Benchmark harness for evaluating color constancy algorithms on standard datasets.

Supports CSV-based datasets with ground-truth illuminants (e.g. Gehler-Shi,
SFU Gray-Ball, NUS-8).  Reports per-method angular error statistics and
generates summary tables.

Usage (CLI)::

    color-constancy-benchmark dataset.csv --image-dir ./images

Usage (API)::

    from color_constancy.benchmark import run_benchmark, load_dataset
    from color_constancy.algorithms import GrayWorldCorrection, WhitePatchCorrection

    entries = load_dataset("dataset.csv", image_dir="./images")
    results = run_benchmark(entries, algorithms={
        "GrayWorld": GrayWorldCorrection(),
        "WhitePatch": WhitePatchCorrection(),
    })
    print(results.to_markdown())
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .algorithms import (
    MSRCR,
    ColorConstancyAlgorithm,
    GrayWorldCorrection,
    MultiScaleRetinex,
    RetinexEnhancement,
    SpatialColorCorrection,
    VonKriesAdaptation,
    WhitePatchCorrection,
)
from .io import load_image
from .metrics import angular_error


@dataclass
class DatasetEntry:
    """A single entry from a color constancy evaluation dataset.

    Attributes
    ----------
    image_path:
        Path to the source image file.
    illuminant:
        Ground-truth illuminant, shape ``(3,)``, already normalised.
    metadata:
        Optional extra fields from the dataset row (split, annotations, etc.).
    """

    image_path: Path
    illuminant: np.ndarray
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class AlgorithmResult:
    """Per-image result for a single algorithm."""

    image_name: str
    angular_error: float
    estimated_illuminant: np.ndarray


@dataclass
class BenchmarkReport:
    """Aggregated benchmark results for multiple algorithms."""

    algorithm_name: str
    errors: list[float]
    results: list[AlgorithmResult] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return float(np.mean(self.errors)) if self.errors else float("nan")

    @property
    def median(self) -> float:
        return float(np.median(self.errors)) if self.errors else float("nan")

    @property
    def trimean(self) -> float:
        """Trimean (Tukey): (Q1 + 2*median + Q3) / 4."""
        if not self.errors:
            return float("nan")
        q1, med, q3 = np.percentile(self.errors, [25, 50, 75])
        return float((q1 + 2 * med + q3) / 4.0)

    @property
    def worst_5(self) -> float:
        """Mean of the worst 5% of errors."""
        if not self.errors:
            return float("nan")
        cutoff = np.percentile(self.errors, 95)
        worst = [e for e in self.errors if e >= cutoff]
        return float(np.mean(worst)) if worst else float("nan")

    @property
    def best_25(self) -> float:
        """Mean of the best 25% of errors."""
        if not self.errors:
            return float("nan")
        cutoff = np.percentile(self.errors, 25)
        best = [e for e in self.errors if e <= cutoff]
        return float(np.mean(best)) if best else float("nan")


@dataclass
class BenchmarkResults:
    """Collection of BenchmarkReport objects keyed by algorithm name."""

    reports: dict[str, BenchmarkReport] = field(default_factory=dict)
    num_images: int = 0

    def add_result(self, algo_name: str, image_name: str, error: float, estimated: np.ndarray) -> None:
        if algo_name not in self.reports:
            self.reports[algo_name] = BenchmarkReport(algo_name, [])
        report = self.reports[algo_name]
        report.errors.append(error)
        report.results.append(AlgorithmResult(image_name, error, estimated))

    def summary_table(self) -> str:
        """Return a formatted summary table as a string."""
        header = (f"{'Algorithm':<22} {'Mean':>8} {'Median':>8} "
                   f"{'Trimean':>8} {'Best25':>8} {'Worst5':>8}  N")
        lines = [header, "-" * len(header)]
        for name in sorted(self.reports):
            r = self.reports[name]
            lines.append(
                f"{name:<22} {r.mean:>8.2f} {r.median:>8.2f} "
                f"{r.trimean:>8.2f} {r.best_25:>8.2f} {r.worst_5:>8.2f}  {len(r.errors)}"
            )
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Return a Markdown-formatted summary table."""
        lines = [
            "| Algorithm | Mean | Median | Trimean | Best 25% | Worst 5% | N |",
            "|-----------|---:|-------:|--------:|---------:|---------:|---|",
        ]
        for name in sorted(self.reports):
            r = self.reports[name]
            lines.append(
                f"| {name} | {r.mean:.2f} | {r.median:.2f} | "
                f"{r.trimean:.2f} | {r.best_25:.2f} | {r.worst_5:.2f} | {len(r.errors)} |"
            )
        return "\n".join(lines)

    def to_csv(self) -> str:
        """Return results as CSV with per-image per-algorithm rows."""
        rows = ["algorithm,image,angular_error"]
        for name in sorted(self.reports):
            for r in self.reports[name].results:
                rows.append(f"{name},{r.image_name},{r.angular_error:.4f}")
        return "\n".join(rows)


def load_dataset(
    csv_path: str,
    image_dir: str = ".",
    illuminant_cols: tuple[str, str, str] = ("r", "g", "b"),
    image_col: str = "filename",
    normalise: bool = True,
) -> list[DatasetEntry]:
    """Load a color constancy dataset from a CSV file.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.
    image_dir:
        Directory containing the images (prepended to image_col values).
    illuminant_cols:
        Names of the three columns containing the ground-truth illuminant
        (R, G, B).
    image_col:
        Name of the column containing the image filename.
    normalise:
        If True (default), normalise the illuminant to unit sum (chromaticity).

    Returns
    -------
    list[DatasetEntry]
    """
    image_dir_path = Path(image_dir)
    entries: list[DatasetEntry] = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            meta = {k: v for k, v in row.items()
                    if k not in (*illuminant_cols, image_col)}
            try:
                illum = np.array([
                    float(row[illuminant_cols[0]]),
                    float(row[illuminant_cols[1]]),
                    float(row[illuminant_cols[2]]),
                ], dtype=np.float64)
            except (KeyError, ValueError) as exc:
                print(f"Warning: skipping row (bad illuminant): {exc}", file=sys.stderr)
                continue

            if normalise and illum.sum() > 0:
                illum = illum / illum.sum()

            image_name = row[image_col]
            image_path = image_dir_path / image_name

            entries.append(DatasetEntry(
                image_path=image_path,
                illuminant=illum.astype(np.float32),
                metadata=meta,
            ))

    return entries


def run_benchmark(
    entries: list[DatasetEntry],
    algorithms: dict[str, ColorConstancyAlgorithm] | None = None,
) -> BenchmarkResults:
    """Run a benchmark across all dataset *entries* using *algorithms*.

    Parameters
    ----------
    entries:
        Dataset entries loaded via :func:`load_dataset`.
    algorithms:
        Mapping of algorithm display name → algorithm instance.  Defaults to
        all built-in algorithms if not provided.

    Returns
    -------
    BenchmarkResults
    """
    if algorithms is None:
        algorithms = _default_algorithms()

    results = BenchmarkResults(num_images=len(entries))

    for entry in entries:
        try:
            image = load_image(str(entry.image_path))
        except (FileNotFoundError, ValueError) as exc:
            print(f"Warning: skipping {entry.image_path}: {exc}", file=sys.stderr)
            continue

        img_f = image.astype(np.float32) / 255.0

        for name, algo in algorithms.items():
            try:
                if hasattr(algo, "estimate_illuminant"):
                    estimated = algo.estimate_illuminant(img_f)
                else:
                    # For pipeline/retinex methods, estimate from the processed output.
                    processed = algo.process(img_f)
                    estimated = processed.mean(axis=(0, 1)).astype(np.float32)
                    total = estimated.sum()
                    if total > 0:
                        estimated = estimated / total

                error = angular_error(estimated.astype(np.float64), entry.illuminant.astype(np.float64))
                results.add_result(name, entry.image_path.name, error, estimated)
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: {name} failed on {entry.image_path.name}: {exc}", file=sys.stderr)

    return results


def _default_algorithms() -> dict[str, ColorConstancyAlgorithm]:
    """Return the default suite of built-in algorithms for benchmarking."""
    return {
        "GrayWorld": GrayWorldCorrection(),
        "WhitePatch": WhitePatchCorrection(),
        "VonKries": VonKriesAdaptation(),
        "Retinex(SSR)": RetinexEnhancement(),
        "MSR": MultiScaleRetinex(),
        "MSRCR": MSRCR(),
        "Spatial": SpatialColorCorrection(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _create_benchmark_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark color constancy algorithms on a dataset.",
    )
    parser.add_argument("csv_path", help="Path to the dataset CSV file.")
    parser.add_argument(
        "--image-dir", default=".",
        help="Directory containing the images (prepended to filenames).",
    )
    parser.add_argument(
        "--illuminant-cols", default="r,g,b",
        help="Comma-separated names of the illuminant columns (default: r,g,b).",
    )
    parser.add_argument(
        "--image-col", default="filename",
        help="Name of the column holding the image filename.",
    )
    parser.add_argument(
        "--no-normalise", action="store_true",
        help="Do not normalise illuminants to unit sum.",
    )
    parser.add_argument(
        "--format", choices=["table", "markdown", "csv"], default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--output", help="Write results to a file instead of stdout.",
    )
    parser.add_argument(
        "--method", nargs="*",
        help="Subset of algorithms to benchmark (default: all).",
    )
    return parser


def main() -> None:
    """Entry point for the ``color-constancy-benchmark`` console script."""
    parser = _create_benchmark_parser()
    args = parser.parse_args()

    illum_cols = tuple(args.illuminant_cols.split(","))
    if len(illum_cols) != 3:
        print("Error: --illuminant-cols must have exactly 3 column names.", file=sys.stderr)
        sys.exit(1)

    entries = load_dataset(
        args.csv_path,
        image_dir=args.image_dir,
        illuminant_cols=illum_cols,  # type: ignore[arg-type]
        image_col=args.image_col,
        normalise=not args.no_normalise,
    )
    if not entries:
        print("Error: no entries loaded from dataset.", file=sys.stderr)
        sys.exit(1)

    algorithms = _default_algorithms()
    if args.method:
        algorithms = {k: v for k, v in algorithms.items() if k in args.method}
        if not algorithms:
            print(f"Error: no matching algorithms found for {args.method}. "
                  f"Available: {list(_default_algorithms())}", file=sys.stderr)
            sys.exit(1)

    print(f"Benchmarking {len(entries)} images with {len(algorithms)} algorithms ...", file=sys.stderr)
    results = run_benchmark(entries, algorithms)

    if args.format == "table":
        output = results.summary_table()
    elif args.format == "markdown":
        output = results.to_markdown()
    else:
        output = results.to_csv()

    if args.output:
        Path(args.output).write_text(output + "\n")
        print(f"Results written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
