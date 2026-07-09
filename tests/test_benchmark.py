"""Tests for the benchmark harness."""

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest

from color_constancy.benchmark import (
    BenchmarkReport,
    BenchmarkResults,
    DatasetEntry,
    load_dataset,
    run_benchmark,
)
from color_constancy.algorithms import GrayWorldCorrection, WhitePatchCorrection
from color_constancy.io import save_image


@pytest.fixture()
def tmp_dataset(tmp_path, random_image):
    """Create a minimal dataset: CSV + two synthetic images."""
    # Save two synthetic images with known simple content.
    neutral = np.full((32, 32, 3), 128, dtype=np.uint8)
    red = np.full((32, 32, 3), [200, 100, 100], dtype=np.uint8)

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    save_image(neutral, str(img_dir / "neutral.png"))
    save_image(red, str(img_dir / "red.png"))

    # CSV with ground-truth illuminants.
    csv_path = tmp_path / "dataset.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "r", "g", "b"])
        writer.writerow(["neutral.png", 1.0, 1.0, 1.0])
        writer.writerow(["red.png", 2.0, 1.0, 1.0])

    return csv_path, img_dir


def test_load_dataset(tmp_dataset):
    csv_path, img_dir = tmp_dataset
    entries = load_dataset(str(csv_path), image_dir=str(img_dir))
    assert len(entries) == 2
    assert entries[0].image_path.name == "neutral.png"
    assert entries[1].image_path.name == "red.png"
    # Check normalisation: (1,1,1) → (1/3, 1/3, 1/3)
    np.testing.assert_allclose(entries[0].illuminant, [1 / 3, 1 / 3, 1 / 3], atol=1e-6)


def test_load_dataset_no_normalise(tmp_dataset):
    csv_path, img_dir = tmp_dataset
    entries = load_dataset(str(csv_path), image_dir=str(img_dir), normalise=False)
    np.testing.assert_allclose(entries[0].illuminant, [1.0, 1.0, 1.0], atol=1e-6)


def test_run_benchmark_basic(tmp_dataset):
    csv_path, img_dir = tmp_dataset
    entries = load_dataset(str(csv_path), image_dir=str(img_dir))
    results = run_benchmark(entries, algorithms={
        "GrayWorld": GrayWorldCorrection(),
        "WhitePatch": WhitePatchCorrection(),
    })
    assert results.num_images == 2
    assert "GrayWorld" in results.reports
    assert "WhitePatch" in results.reports
    assert len(results.reports["GrayWorld"].errors) == 2


def test_benchmark_report_stats():
    report = BenchmarkReport("TestAlgo", [1.0, 2.0, 3.0, 4.0, 5.0])
    assert report.mean == 3.0
    assert report.median == 3.0
    assert report.trimean == 3.0


def test_benchmark_results_summary_table(tmp_dataset):
    csv_path, img_dir = tmp_dataset
    entries = load_dataset(str(csv_path), image_dir=str(img_dir))
    results = run_benchmark(entries, algorithms={"GrayWorld": GrayWorldCorrection()})
    table = results.summary_table()
    assert "GrayWorld" in table
    assert "Mean" in table


def test_benchmark_results_to_markdown(tmp_dataset):
    csv_path, img_dir = tmp_dataset
    entries = load_dataset(str(csv_path), image_dir=str(img_dir))
    results = run_benchmark(entries, algorithms={"GrayWorld": GrayWorldCorrection()})
    md = results.to_markdown()
    assert "| Algorithm |" in md
    assert "GrayWorld" in md


def test_benchmark_results_to_csv(tmp_dataset):
    csv_path, img_dir = tmp_dataset
    entries = load_dataset(str(csv_path), image_dir=str(img_dir))
    results = run_benchmark(entries, algorithms={"GrayWorld": GrayWorldCorrection()})
    csv_out = results.to_csv()
    assert "algorithm,image,angular_error" in csv_out


def test_run_benchmark_with_pipeline_algorithms(tmp_dataset):
    """Algorithms without estimate_illuminant should fall back to output mean."""
    from color_constancy.algorithms import MultiScaleRetinex
    csv_path, img_dir = tmp_dataset
    entries = load_dataset(str(csv_path), image_dir=str(img_dir))
    results = run_benchmark(entries, algorithms={"MSR": MultiScaleRetinex()})
    assert "MSR" in results.reports
    assert len(results.reports["MSR"].errors) == 2
    assert all(np.isfinite(e) for e in results.reports["MSR"].errors)