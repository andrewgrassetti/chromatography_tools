"""Tests for the Chromatogram class."""

from __future__ import annotations

import numpy as np
import pytest

from chromatography.core import Chromatogram, Peak


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gaussian(time: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.exp(-((time - b) ** 2) / (2 * c ** 2))


def _single_peak_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    time = np.linspace(0, 20, 200)
    intensity = _make_gaussian(time, 100, 10, 2) + rng.normal(scale=5, size=len(time))
    return time, intensity


def _two_peak_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(100)
    time = np.linspace(0, 20, 800)
    peak1 = _make_gaussian(time, 90, 9.5, 0.25)
    peak2 = _make_gaussian(time, 70, 10.6, 0.22)
    intensity = peak1 + peak2 + rng.normal(scale=4, size=len(time))
    return time, intensity


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_basic(self):
        t, i = _single_peak_data()
        ch = Chromatogram(t, i)
        assert len(ch.time) == len(t)
        assert ch.smoothed is None
        assert ch.peaks == []

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            Chromatogram([1, 2, 3], [1, 2])


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

class TestSmoothing:
    def test_smooth_reduces_noise(self):
        t, i = _single_peak_data()
        ch = Chromatogram(t, i).smooth(window=11, poly=3)
        assert ch.smoothed is not None
        # smoothed signal should have smaller std deviation from a clean Gaussian
        clean = _make_gaussian(t, 100, 10, 2)
        raw_rmse = np.sqrt(np.mean((i - clean) ** 2))
        smooth_rmse = np.sqrt(np.mean((ch.smoothed - clean) ** 2))
        assert smooth_rmse < raw_rmse

    def test_smooth_fallback_for_short_series(self):
        ch = Chromatogram([1, 2, 3], [10, 20, 10]).smooth(window=11, poly=3)
        np.testing.assert_array_equal(ch.smoothed, ch.intensity)


# ---------------------------------------------------------------------------
# Gaussian fitting
# ---------------------------------------------------------------------------

class TestGaussianFit:
    def test_fit_recovers_params(self):
        t, i = _single_peak_data()
        ch = Chromatogram(t, i).smooth().fit_gaussian()
        assert ch.fit_params is not None
        assert abs(ch.fit_params["b"] - 10) < 1  # peak center near 10

    def test_fit_without_smooth_raises(self):
        t, i = _single_peak_data()
        with pytest.raises(RuntimeError, match="Smooth"):
            Chromatogram(t, i).fit_gaussian()


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

class TestBaseline:
    def test_min_baseline(self):
        t, i = _single_peak_data()
        ch = Chromatogram(t, i).baseline("min")
        assert ch.baseline_subtracted is not None
        assert np.nanmin(ch.baseline_subtracted) >= -1e-10  # approximately >= 0

    def test_median_baseline(self):
        ch = Chromatogram([0, 1, 2], [10, 50, 10]).baseline("median")
        np.testing.assert_allclose(ch.baseline_subtracted, [0, 40, 0])

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            Chromatogram([0, 1], [1, 2]).baseline("magic")


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------

class TestAUC:
    def test_auc_positive(self):
        t, i = _single_peak_data()
        ch = Chromatogram(t, i)
        assert ch.auc() > 0

    def test_auc_smoothed(self):
        t, i = _single_peak_data()
        ch = Chromatogram(t, i).smooth()
        auc_smooth = ch.auc(use="smoothed")
        auc_raw = ch.auc(use="raw")
        assert abs(auc_smooth - auc_raw) / auc_raw < 0.1  # within 10%


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------

class TestPeakDetection:
    def test_single_peak_found(self):
        t, i = _single_peak_data()
        ch = Chromatogram(t, i).smooth().find_peaks()
        assert len(ch.peaks) >= 1
        # Main peak should be near t=10
        main_peak = max(ch.peaks, key=lambda p: p.height)
        assert abs(main_peak.time - 10) < 1

    def test_two_peaks_found(self):
        t, i = _two_peak_data()
        ch = Chromatogram(t, i).smooth().find_peaks()
        assert len(ch.peaks) == 2

    def test_no_peaks_below_threshold(self):
        t, i = _single_peak_data()
        ch = Chromatogram(t, i).smooth().find_peaks(min_height=1e6)
        assert len(ch.peaks) == 0

    def test_short_series(self):
        ch = Chromatogram([1, 2], [10, 20]).find_peaks()
        assert ch.peaks == []


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_integration_areas_sum_to_100(self):
        t, i = _two_peak_data()
        ch = Chromatogram(t, i).smooth().find_peaks()
        ch.integrate_peaks()
        total_pct = sum(p.rel_area_pct for p in ch.peaks if p.rel_area_pct is not None)
        assert abs(total_pct - 100) < 0.01

    def test_integration_without_peaks_raises(self):
        t, i = _single_peak_data()
        with pytest.raises(RuntimeError, match="No peaks"):
            Chromatogram(t, i).integrate_peaks()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_with_peaks(self):
        t, i = _two_peak_data()
        ch = Chromatogram(t, i).smooth().find_peaks().integrate_peaks()
        summary = ch.summarize_peaks()
        assert "Detected 2 peak(s)" in summary
        assert "Area:" in summary

    def test_summary_no_peaks(self):
        t, i = _single_peak_data()
        ch = Chromatogram(t, i)
        assert ch.summarize_peaks() == "No peaks detected."
