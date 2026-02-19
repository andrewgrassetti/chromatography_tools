"""Tests for the chromatography OOP class hierarchy."""

from __future__ import annotations

import numpy as np
import pytest

from chromatography.core import (
    BaseChromatogram,
    Chromatogram,
    GCChromatogram,
    HPLCChromatogram,
    IonChromatogram,
    Peak,
    SECChromatogram,
)


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
# OOP hierarchy
# ---------------------------------------------------------------------------

class TestHierarchy:
    def test_base_is_abstract(self):
        with pytest.raises(TypeError):
            BaseChromatogram([1, 2], [3, 4])  # type: ignore[abstract]

    def test_chromatogram_alias(self):
        assert Chromatogram is HPLCChromatogram

    def test_all_subclasses_instantiate(self):
        t, i = _single_peak_data()
        for cls in (HPLCChromatogram, GCChromatogram, SECChromatogram, IonChromatogram):
            ch = cls(t, i)
            assert isinstance(ch, BaseChromatogram)

    def test_technique_names(self):
        t, i = [0, 1], [0, 1]
        assert HPLCChromatogram(t, i).technique_name == "HPLC"
        assert GCChromatogram(t, i).technique_name == "GC"
        assert SECChromatogram(t, i).technique_name == "SEC/GPC"
        assert IonChromatogram(t, i).technique_name == "IC"

    def test_axis_labels(self):
        t, i = [0, 1], [0, 1]
        hplc = HPLCChromatogram(t, i)
        assert "Retention Time" in hplc.x_axis_label
        assert "Absorbance" in hplc.y_axis_label

        gc = GCChromatogram(t, i)
        assert "Retention Time" in gc.x_axis_label

        sec = SECChromatogram(t, i)
        assert "Elution Volume" in sec.x_axis_label

        ic = IonChromatogram(t, i)
        assert "Conductivity" in ic.y_axis_label

    def test_technique_specific_defaults(self):
        t, i = [0, 1], [0, 1]
        assert GCChromatogram(t, i).default_smooth_window == 7
        assert HPLCChromatogram(t, i).default_smooth_window == 11
        assert SECChromatogram(t, i).default_smooth_window == 15
        assert IonChromatogram(t, i).default_smooth_window == 15

        assert GCChromatogram(t, i).default_min_height_frac == 0.05
        assert HPLCChromatogram(t, i).default_min_height_frac == 0.05


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_basic(self):
        t, i = _single_peak_data()
        ch = HPLCChromatogram(t, i)
        assert len(ch.time) == len(t)
        assert ch.smoothed is None
        assert ch.peaks == []

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            HPLCChromatogram([1, 2, 3], [1, 2])


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

class TestSmoothing:
    def test_smooth_reduces_noise(self):
        t, i = _single_peak_data()
        ch = HPLCChromatogram(t, i).smooth(window=11, poly=3)
        assert ch.smoothed is not None
        clean = _make_gaussian(t, 100, 10, 2)
        raw_rmse = np.sqrt(np.mean((i - clean) ** 2))
        smooth_rmse = np.sqrt(np.mean((ch.smoothed - clean) ** 2))
        assert smooth_rmse < raw_rmse

    def test_smooth_fallback_for_short_series(self):
        ch = HPLCChromatogram([1, 2, 3], [10, 20, 10]).smooth(window=11, poly=3)
        np.testing.assert_array_equal(ch.smoothed, ch.intensity)

    def test_smooth_uses_technique_defaults(self):
        t, i = _single_peak_data()
        gc = GCChromatogram(t, i).smooth()  # window=7 by default
        assert gc.smoothed is not None

    def test_smooth_overrides_defaults(self):
        t, i = _single_peak_data()
        gc = GCChromatogram(t, i).smooth(window=15, poly=2)
        assert gc.smoothed is not None


# ---------------------------------------------------------------------------
# Gaussian fitting
# ---------------------------------------------------------------------------

class TestGaussianFit:
    def test_fit_recovers_params(self):
        t, i = _single_peak_data()
        ch = HPLCChromatogram(t, i).smooth().fit_gaussian()
        assert ch.fit_params is not None
        assert abs(ch.fit_params["b"] - 10) < 1

    def test_fit_without_smooth_raises(self):
        t, i = _single_peak_data()
        with pytest.raises(RuntimeError, match="Smooth"):
            HPLCChromatogram(t, i).fit_gaussian()


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

class TestBaseline:
    def test_min_baseline(self):
        t, i = _single_peak_data()
        ch = HPLCChromatogram(t, i).baseline("min")
        assert ch.baseline_subtracted is not None
        assert np.nanmin(ch.baseline_subtracted) >= -1e-10

    def test_median_baseline(self):
        ch = HPLCChromatogram([0, 1, 2], [10, 50, 10]).baseline("median")
        np.testing.assert_allclose(ch.baseline_subtracted, [0, 40, 0])

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            HPLCChromatogram([0, 1], [1, 2]).baseline("magic")


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------

class TestAUC:
    def test_auc_positive(self):
        t, i = _single_peak_data()
        assert HPLCChromatogram(t, i).auc() > 0

    def test_auc_smoothed(self):
        t, i = _single_peak_data()
        ch = HPLCChromatogram(t, i).smooth()
        auc_smooth = ch.auc(use="smoothed")
        auc_raw = ch.auc(use="raw")
        assert abs(auc_smooth - auc_raw) / auc_raw < 0.1

    def test_trapezoid_compat(self):
        """_trapezoid helper must resolve regardless of NumPy version."""
        from chromatography.core import _trapezoid
        assert callable(_trapezoid)
        # Sanity check: area of a simple triangle (base=2, height=1) == 1.0
        assert _trapezoid([0, 1, 0], [0, 1, 2]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------

class TestPeakDetection:
    def test_single_peak_found(self):
        t, i = _single_peak_data()
        ch = HPLCChromatogram(t, i).smooth().find_peaks()
        assert len(ch.peaks) >= 1
        main_peak = max(ch.peaks, key=lambda p: p.height)
        assert abs(main_peak.time - 10) < 1

    def test_two_peaks_found(self):
        t, i = _two_peak_data()
        ch = HPLCChromatogram(t, i).smooth().find_peaks()
        assert len(ch.peaks) >= 2
        # There should be at least one peak near each true location
        peak_times = [p.time for p in ch.peaks]
        assert any(abs(pt - 9.5) < 0.5 for pt in peak_times)
        assert any(abs(pt - 10.6) < 0.5 for pt in peak_times)

    def test_no_peaks_below_threshold(self):
        t, i = _single_peak_data()
        ch = HPLCChromatogram(t, i).smooth().find_peaks(min_height=1e6)
        assert len(ch.peaks) == 0

    def test_short_series(self):
        ch = HPLCChromatogram([1, 2], [10, 20]).find_peaks()
        assert ch.peaks == []

    def test_gc_finds_peaks(self):
        t, i = _two_peak_data()
        ch = GCChromatogram(t, i).smooth().find_peaks()
        assert len(ch.peaks) >= 2

    def test_sec_finds_peaks(self):
        t, i = _single_peak_data()
        ch = SECChromatogram(t, i).smooth().find_peaks()
        assert len(ch.peaks) >= 1

    def test_prominence_filters_noise(self):
        """High prominence threshold should suppress noise peaks."""
        t, i = _single_peak_data()
        ch = HPLCChromatogram(t, i).smooth().find_peaks(prominence=50)
        assert len(ch.peaks) >= 1
        # Only the main peak should survive a high prominence threshold
        main_peak = max(ch.peaks, key=lambda p: p.height)
        assert abs(main_peak.time - 10) < 1

    def test_distance_parameter(self):
        """Setting a large distance should merge nearby spurious peaks."""
        t, i = _two_peak_data()
        ch = HPLCChromatogram(t, i).smooth().find_peaks(distance=5)
        assert len(ch.peaks) >= 2

    def test_peak_bounds_within_signal(self):
        """Integration bounds (start/end) should be within signal range."""
        t, i = _two_peak_data()
        ch = HPLCChromatogram(t, i).smooth().find_peaks()
        for pk in ch.peaks:
            assert 0 <= pk.start < len(t)
            assert 0 <= pk.end < len(t)
            assert pk.start < pk.end

    def test_overlapping_peaks_split_at_valley(self):
        """Adjacent peaks without baseline separation should not overlap."""
        t, i = _two_peak_data()
        ch = HPLCChromatogram(t, i).smooth().find_peaks()
        assert len(ch.peaks) >= 2
        # Bounds of adjacent peaks should not overlap
        sorted_peaks = sorted(ch.peaks, key=lambda p: p.time)
        for k in range(len(sorted_peaks) - 1):
            assert sorted_peaks[k].end <= sorted_peaks[k + 1].start

    def test_overlapping_peaks_from_csv(self):
        """Peaks from TestFile.CSV should be detected and split properly."""
        import os
        import pandas as pd
        csv_path = os.path.join(os.path.dirname(__file__), "TestFile.CSV")
        df = pd.read_csv(csv_path, header=None)
        df.columns = ["time", "intensity"]
        ch = HPLCChromatogram(
            df["time"].values, df["intensity"].values,
        ).smooth().find_peaks()
        assert len(ch.peaks) >= 2
        # Adjacent peak bounds should meet but not overlap
        sorted_peaks = sorted(ch.peaks, key=lambda p: p.time)
        for k in range(len(sorted_peaks) - 1):
            assert sorted_peaks[k].end <= sorted_peaks[k + 1].start
        # Integration should still sum to 100%
        ch.integrate_peaks()
        total_pct = sum(p.rel_area_pct for p in ch.peaks if p.rel_area_pct is not None)
        assert abs(total_pct - 100) < 0.01

    def test_overlapping_peaks_from_csv2(self):
        """Peaks from TestFile2.CSV should be detected and split properly."""
        import os
        import pandas as pd
        csv_path = os.path.join(os.path.dirname(__file__), "TestFile2.CSV")
        df = pd.read_csv(csv_path, header=None)
        df.columns = ["time", "intensity"]
        ch = HPLCChromatogram(
            df["time"].values, df["intensity"].values,
        ).smooth().find_peaks()
        assert len(ch.peaks) >= 2
        # Adjacent peak bounds should meet but not overlap
        sorted_peaks = sorted(ch.peaks, key=lambda p: p.time)
        for k in range(len(sorted_peaks) - 1):
            assert sorted_peaks[k].end <= sorted_peaks[k + 1].start
        # Integration should still sum to 100%
        ch.integrate_peaks()
        total_pct = sum(p.rel_area_pct for p in ch.peaks if p.rel_area_pct is not None)
        assert abs(total_pct - 100) < 0.01


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_integration_areas_sum_to_100(self):
        t, i = _two_peak_data()
        ch = HPLCChromatogram(t, i).smooth().find_peaks()
        ch.integrate_peaks()
        total_pct = sum(p.rel_area_pct for p in ch.peaks if p.rel_area_pct is not None)
        assert abs(total_pct - 100) < 0.01

    def test_integration_without_peaks_raises(self):
        t, i = _single_peak_data()
        with pytest.raises(RuntimeError, match="No peaks"):
            HPLCChromatogram(t, i).integrate_peaks()


# ---------------------------------------------------------------------------
# SEC molecular-weight calibration
# ---------------------------------------------------------------------------

class TestSECCalibration:
    def test_mw_calibration(self):
        t, i = _single_peak_data()

        def mw_cal(v: np.ndarray) -> np.ndarray:
            return 10 ** (10 - 0.5 * v)

        ch = SECChromatogram(t, i, mw_calibration=mw_cal)
        mw = ch.molecular_weights()
        assert mw is not None
        assert len(mw) == len(t)
        assert mw[0] > mw[-1]  # MW decreases with elution volume

    def test_no_calibration_returns_none(self):
        ch = SECChromatogram([0, 1], [1, 2])
        assert ch.molecular_weights() is None


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_with_peaks(self):
        t, i = _two_peak_data()
        ch = HPLCChromatogram(t, i).smooth().find_peaks().integrate_peaks()
        summary = ch.summarize_peaks()
        assert "peak(s)" in summary
        assert "Area:" in summary

    def test_summary_no_peaks(self):
        t, i = _single_peak_data()
        ch = HPLCChromatogram(t, i)
        assert ch.summarize_peaks() == "No peaks detected."


# ---------------------------------------------------------------------------
# Method chaining
# ---------------------------------------------------------------------------

class TestChaining:
    def test_fluent_interface(self):
        t, i = _two_peak_data()
        ch = (
            HPLCChromatogram(t, i)
            .smooth()
            .find_peaks()
            .integrate_peaks()
        )
        assert len(ch.peaks) >= 2
        assert all(p.rel_area_pct is not None for p in ch.peaks)


# ---------------------------------------------------------------------------
# Manual peak-bound override
# ---------------------------------------------------------------------------

class TestManualBounds:
    def test_set_left_bound_narrows_peak(self):
        """Setting a left bound closer to the apex narrows the integration window."""
        t, i = _single_peak_data()
        ch = HPLCChromatogram(t, i).smooth().find_peaks()
        assert len(ch.peaks) >= 1
        pk = max(ch.peaks, key=lambda p: p.height)
        original_start = pk.start
        # Set left bound to be halfway between original start and apex
        new_left = (t[original_start] + pk.time) / 2
        ch.set_manual_bounds(pk.time, left_time=new_left)
        # Peak should still exist (bound is left of apex)
        matching = [p for p in ch.peaks if abs(p.time - pk.time) < 0.5]
        assert len(matching) == 1
        assert matching[0].start > original_start

    def test_set_right_bound_narrows_peak(self):
        """Setting a right bound closer to the apex narrows the integration window."""
        t, i = _single_peak_data()
        ch = HPLCChromatogram(t, i).smooth().find_peaks()
        pk = max(ch.peaks, key=lambda p: p.height)
        original_end = pk.end
        ch.set_manual_bounds(pk.time, right_time=pk.time + 1.0)
        matching = [p for p in ch.peaks if abs(p.time - pk.time) < 0.5]
        assert len(matching) == 1
        assert matching[0].end <= original_end

    def test_set_both_bounds(self):
        """Both bounds can be set simultaneously."""
        t, i = _single_peak_data()
        ch = HPLCChromatogram(t, i).smooth().find_peaks()
        pk = max(ch.peaks, key=lambda p: p.height)
        ch.set_manual_bounds(
            pk.time, left_time=pk.time - 0.5, right_time=pk.time + 0.5,
        )
        matching = [p for p in ch.peaks if abs(p.time - pk.time) < 0.5]
        assert len(matching) == 1

    def test_invalid_left_bound_removes_peak(self):
        """A left bound beyond the apex should remove the peak."""
        t, i = _single_peak_data()
        ch = HPLCChromatogram(t, i).smooth().find_peaks()
        pk = max(ch.peaks, key=lambda p: p.height)
        original_count = len(ch.peaks)
        # Set left bound *past* the apex
        ch.set_manual_bounds(pk.time, left_time=pk.time + 2.0)
        assert len(ch.peaks) == original_count - 1

    def test_invalid_right_bound_removes_peak(self):
        """A right bound before the apex should remove the peak."""
        t, i = _single_peak_data()
        ch = HPLCChromatogram(t, i).smooth().find_peaks()
        pk = max(ch.peaks, key=lambda p: p.height)
        original_count = len(ch.peaks)
        ch.set_manual_bounds(pk.time, right_time=pk.time - 2.0)
        assert len(ch.peaks) == original_count - 1

    def test_no_peaks_raises(self):
        """Calling without detected peaks should raise RuntimeError."""
        ch = HPLCChromatogram([0, 1, 2], [1, 2, 1])
        with pytest.raises(RuntimeError, match="No peaks"):
            ch.set_manual_bounds(1.0, left_time=0.5)

    def test_no_bounds_raises(self):
        """Calling without any bound should raise ValueError."""
        t, i = _single_peak_data()
        ch = HPLCChromatogram(t, i).smooth().find_peaks()
        with pytest.raises(ValueError, match="left_time or right_time"):
            ch.set_manual_bounds(10.0)

    def test_reintegration_after_manual_bounds(self):
        """Integration should reflect the manually adjusted bounds."""
        t, i = _two_peak_data()
        ch = HPLCChromatogram(t, i).smooth().find_peaks().integrate_peaks()
        original_areas = {p.time: p.rel_area_pct for p in ch.peaks}

        # Re-detect and narrow the first peak
        ch2 = HPLCChromatogram(t, i).smooth().find_peaks()
        pk = min(ch2.peaks, key=lambda p: p.time)
        target_right = pk.time + 0.15
        ch2.set_manual_bounds(pk.time, right_time=target_right)
        # Verify the bound was actually set near the requested time
        updated_pk = min(ch2.peaks, key=lambda p: p.time)
        assert t[updated_pk.end] <= target_right + (t[1] - t[0])
        ch2.integrate_peaks()

        # Areas should have changed
        new_areas = {p.time: p.rel_area_pct for p in ch2.peaks}
        assert new_areas != original_areas

    def test_method_chaining(self):
        """set_manual_bounds returns self for fluent chaining."""
        t, i = _single_peak_data()
        ch = HPLCChromatogram(t, i).smooth().find_peaks()
        pk = ch.peaks[0]
        result = ch.set_manual_bounds(pk.time, left_time=pk.time - 1.0)
        assert result is ch

    def test_finds_nearest_peak(self):
        """Should select the peak closest to the given time."""
        t, i = _two_peak_data()
        ch = HPLCChromatogram(t, i).smooth().find_peaks()
        sorted_peaks = sorted(ch.peaks, key=lambda p: p.time)
        first_pk = sorted_peaks[0]
        # Use a time closer to the first peak
        ch.set_manual_bounds(first_pk.time + 0.01, left_time=first_pk.time - 0.2)
        matching = [p for p in ch.peaks if abs(p.time - first_pk.time) < 0.5]
        assert len(matching) == 1

    def test_auto_detect_left_bound(self):
        """Passing a time left of the apex as left_time narrows the left side."""
        t, i = _single_peak_data()
        ch = HPLCChromatogram(t, i).smooth().find_peaks()
        pk = max(ch.peaks, key=lambda p: p.height)
        original_start = pk.start
        original_end = pk.end
        # Click halfway between the current left bound and the apex
        clicked_time = (t[original_start] + pk.time) / 2
        ch.set_manual_bounds(pk.time, left_time=clicked_time)
        matching = [p for p in ch.peaks if abs(p.time - pk.time) < 0.5]
        assert len(matching) == 1
        assert matching[0].start > original_start
        assert matching[0].end == original_end

    def test_auto_detect_right_bound(self):
        """Passing a time right of the apex as right_time narrows the right side."""
        t, i = _single_peak_data()
        ch = HPLCChromatogram(t, i).smooth().find_peaks()
        pk = max(ch.peaks, key=lambda p: p.height)
        original_start = pk.start
        original_end = pk.end
        clicked_time = pk.time + 1.5
        ch.set_manual_bounds(pk.time, right_time=clicked_time)
        matching = [p for p in ch.peaks if abs(p.time - pk.time) < 0.5]
        assert len(matching) == 1
        assert matching[0].start == original_start
        assert matching[0].end < original_end
