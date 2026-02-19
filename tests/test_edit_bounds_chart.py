"""Tests for Plotly figure configuration when edit-bounds mode is active.

These tests validate that the chart is properly configured for interactive
peak-bound selection: drag disabled (dragmode=False), single-click
support (clickmode="event+select"), invisible markers on line traces
for reliable point-click detection, crosshair cursor CSS, and
simultaneous overlay-mode bound updates.
"""

from __future__ import annotations

import io
import json

import numpy as np
import plotly.graph_objects as go

from chromatography.core import HPLCChromatogram


def _build_sample_figure() -> go.Figure:
    """Return a simple Plotly figure mimicking the app's chromatogram chart."""
    t = np.linspace(0, 20, 200)
    y = 100 * np.exp(-((t - 10) ** 2) / 8)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=y, mode="lines",
        name="sample", line=dict(color="#440154", width=2),
    ))
    # Add a fill trace (like the peak-bounds shading) that should NOT get markers.
    fig.add_trace(go.Scatter(
        x=t[:50], y=y[:50],
        fill="tozeroy", fillcolor="rgba(68,1,84,0.15)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    return fig


# The CSS snippet that app.py injects for crosshair cursors in edit mode.
CROSSHAIR_CSS = "<style>.nsewdrag { cursor: crosshair !important; }</style>"


def _apply_edit_bounds_config(fig: go.Figure) -> None:
    """Apply the same configuration that app.py applies when edit_bounds is True."""
    fig.update_layout(dragmode=False, clickmode="event+select")
    for trace_data in fig.data:
        if getattr(trace_data, "mode", None) == "lines":
            trace_data.update(
                mode="lines+markers",
                marker=dict(size=8, opacity=0),
            )


class TestEditBoundsChartConfig:
    def test_dragmode_is_disabled(self):
        """dragmode should be False so click-and-drag selection is disabled."""
        fig = _build_sample_figure()
        _apply_edit_bounds_config(fig)
        assert fig.layout.dragmode is False

    def test_clickmode_is_event_select(self):
        """clickmode should be 'event+select' so single clicks register."""
        fig = _build_sample_figure()
        _apply_edit_bounds_config(fig)
        assert fig.layout.clickmode == "event+select"

    def test_line_traces_get_invisible_markers(self):
        """Main line traces should gain invisible markers for click snapping."""
        fig = _build_sample_figure()
        _apply_edit_bounds_config(fig)
        line_trace = fig.data[0]
        assert line_trace.mode == "lines+markers"
        assert line_trace.marker.size == 8
        assert line_trace.marker.opacity == 0

    def test_fill_traces_unchanged(self):
        """Fill (shaded-area) traces should not be altered."""
        fig = _build_sample_figure()
        _apply_edit_bounds_config(fig)
        fill_trace = fig.data[1]
        # The fill trace has line.width=0, not mode="lines", so it should be untouched.
        assert fill_trace.fill == "tozeroy"
        assert fill_trace.mode is None  # mode was never set to "lines"
        assert fill_trace.marker.size is None  # no marker added

    def test_crosshair_css_targets_nsewdrag(self):
        """The injected CSS should set crosshair cursor on .nsewdrag."""
        assert "cursor: crosshair" in CROSSHAIR_CSS
        assert ".nsewdrag" in CROSSHAIR_CSS
        assert "!important" in CROSSHAIR_CSS


# ------------------------------------------------------------------
# Helpers that replicate the overlay click-handling logic from app.py
# ------------------------------------------------------------------

def _make_gaussian(time, a, b, c):
    return a * np.exp(-((time - b) ** 2) / (2 * c ** 2))


def _make_chromatogram_entry(filename, time, intensity):
    """Build a chromatogram entry dict like app.py creates."""
    ch = HPLCChromatogram(time, intensity)
    ch.smooth()
    ch.find_peaks()
    if ch.peaks:
        ch.integrate_peaks()
    return {"filename": filename, "label": filename, "chrom": ch}


def _simulate_overlay_click(chromatograms, clicked_time, manual_bounds):
    """Replicate the overlay-mode click handler from app.py."""
    for entry in chromatograms:
        ch = entry["chrom"]
        if not ch.peaks:
            continue
        nearest_pk = min(ch.peaks, key=lambda pk: abs(pk.time - clicked_time))
        key = (entry["filename"], nearest_pk.time)
        existing = manual_bounds.get(key, {})
        if clicked_time < nearest_pk.time:
            existing["left_time"] = clicked_time
        else:
            existing["right_time"] = clicked_time
        manual_bounds[key] = existing


def _simulate_single_click(chromatograms, clicked_time, manual_bounds,
                           clicked_curve=None, trace_to_chrom_idx=None):
    """Replicate the separate-panels click handler from app.py.

    When *clicked_curve* and *trace_to_chrom_idx* are provided the
    function targets only the chromatogram whose line trace was clicked.
    Otherwise it falls back to finding the single nearest peak across
    all chromatograms.
    """
    target_entry = None
    if clicked_curve is not None and trace_to_chrom_idx and clicked_curve in trace_to_chrom_idx:
        target_entry = chromatograms[trace_to_chrom_idx[clicked_curve]]

    if target_entry is not None:
        ch = target_entry["chrom"]
        if ch.peaks:
            nearest_pk = min(ch.peaks, key=lambda pk: abs(pk.time - clicked_time))
            key = (target_entry["filename"], nearest_pk.time)
            existing = manual_bounds.get(key, {})
            if clicked_time < nearest_pk.time:
                existing["left_time"] = clicked_time
            else:
                existing["right_time"] = clicked_time
            manual_bounds[key] = existing
    else:
        best_entry = None
        best_peak = None
        best_dist = float("inf")
        for entry in chromatograms:
            ch = entry["chrom"]
            for pk in ch.peaks:
                d = abs(pk.time - clicked_time)
                if d < best_dist:
                    best_dist = d
                    best_peak = pk
                    best_entry = entry
        if best_peak is not None and best_entry is not None:
            key = (best_entry["filename"], best_peak.time)
            existing = manual_bounds.get(key, {})
            if clicked_time < best_peak.time:
                existing["left_time"] = clicked_time
            else:
                existing["right_time"] = clicked_time
            manual_bounds[key] = existing


class TestOverlayBoundsClick:
    """Validate that overlay-mode clicks set bounds on all chromatograms."""

    def _two_chromatogram_entries(self):
        """Create two chromatogram entries with peaks at similar positions."""
        rng = np.random.default_rng(42)
        t = np.linspace(0, 20, 400)
        y1 = _make_gaussian(t, 100, 10, 1.5) + rng.normal(scale=2, size=len(t))
        y2 = _make_gaussian(t, 80, 10.2, 1.3) + rng.normal(scale=2, size=len(t))
        entry1 = _make_chromatogram_entry("file1.csv", t, y1)
        entry2 = _make_chromatogram_entry("file2.csv", t, y2)
        return [entry1, entry2]

    def test_overlay_click_sets_bounds_for_all_chromatograms(self):
        """A click in overlay mode should create bound entries for each file."""
        entries = self._two_chromatogram_entries()
        manual_bounds: dict = {}
        _simulate_overlay_click(entries, 8.5, manual_bounds)
        filenames_with_bounds = {k[0] for k in manual_bounds}
        assert "file1.csv" in filenames_with_bounds
        assert "file2.csv" in filenames_with_bounds

    def test_overlay_click_left_of_apex_sets_left_bound(self):
        """A click left of peaks should set left_time for all chromatograms."""
        entries = self._two_chromatogram_entries()
        manual_bounds: dict = {}
        _simulate_overlay_click(entries, 8.5, manual_bounds)
        for bounds in manual_bounds.values():
            assert "left_time" in bounds
            assert bounds["left_time"] == 8.5

    def test_overlay_click_right_of_apex_sets_right_bound(self):
        """A click right of peaks should set right_time for all chromatograms."""
        entries = self._two_chromatogram_entries()
        manual_bounds: dict = {}
        _simulate_overlay_click(entries, 11.5, manual_bounds)
        for bounds in manual_bounds.values():
            assert "right_time" in bounds
            assert bounds["right_time"] == 11.5

    def test_single_mode_click_affects_only_one_chromatogram(self):
        """In separate panels mode, only the nearest peak gets a bound."""
        entries = self._two_chromatogram_entries()
        manual_bounds: dict = {}
        _simulate_single_click(entries, 8.5, manual_bounds)
        assert len(manual_bounds) == 1

    def test_overlay_click_skips_chromatograms_without_peaks(self):
        """Chromatograms with no peaks should be silently skipped."""
        entries = self._two_chromatogram_entries()
        # Clear peaks from the second chromatogram
        entries[1]["chrom"].peaks = []
        manual_bounds: dict = {}
        _simulate_overlay_click(entries, 8.5, manual_bounds)
        filenames_with_bounds = {k[0] for k in manual_bounds}
        assert "file1.csv" in filenames_with_bounds
        assert "file2.csv" not in filenames_with_bounds


class TestSeparatePanelBoundsClick:
    """Validate that separate-panel clicks only affect the clicked panel's chromatogram."""

    def _two_chromatogram_entries(self):
        """Create two chromatogram entries with peaks at distinct positions."""
        rng = np.random.default_rng(42)
        t = np.linspace(0, 20, 400)
        y1 = _make_gaussian(t, 100, 8, 1.5) + rng.normal(scale=2, size=len(t))
        y2 = _make_gaussian(t, 80, 12, 1.3) + rng.normal(scale=2, size=len(t))
        entry1 = _make_chromatogram_entry("file1.csv", t, y1)
        entry2 = _make_chromatogram_entry("file2.csv", t, y2)
        return [entry1, entry2]

    def test_click_on_second_panel_affects_only_second_chromatogram(self):
        """Clicking on the second panel should only set bounds for file2."""
        entries = self._two_chromatogram_entries()
        manual_bounds: dict = {}
        # trace 0 → chromatogram 0, trace 1 → chromatogram 1
        trace_map = {0: 0, 1: 1}
        _simulate_single_click(entries, 11.0, manual_bounds,
                               clicked_curve=1, trace_to_chrom_idx=trace_map)
        filenames_with_bounds = {k[0] for k in manual_bounds}
        assert "file2.csv" in filenames_with_bounds
        assert "file1.csv" not in filenames_with_bounds

    def test_click_on_first_panel_affects_only_first_chromatogram(self):
        """Clicking on the first panel should only set bounds for file1."""
        entries = self._two_chromatogram_entries()
        manual_bounds: dict = {}
        trace_map = {0: 0, 1: 1}
        _simulate_single_click(entries, 7.0, manual_bounds,
                               clicked_curve=0, trace_to_chrom_idx=trace_map)
        filenames_with_bounds = {k[0] for k in manual_bounds}
        assert "file1.csv" in filenames_with_bounds
        assert "file2.csv" not in filenames_with_bounds

    def test_fallback_without_curve_number(self):
        """Without curve_number, falls back to nearest-peak-across-all behaviour."""
        entries = self._two_chromatogram_entries()
        manual_bounds: dict = {}
        _simulate_single_click(entries, 7.0, manual_bounds)
        assert len(manual_bounds) == 1

    def test_click_left_of_apex_sets_left_bound(self):
        """A click left of the panel's peak sets left_time."""
        entries = self._two_chromatogram_entries()
        manual_bounds: dict = {}
        trace_map = {0: 0, 1: 1}
        _simulate_single_click(entries, 11.0, manual_bounds,
                               clicked_curve=1, trace_to_chrom_idx=trace_map)
        for bounds in manual_bounds.values():
            assert "left_time" in bounds
            assert bounds["left_time"] == 11.0

    def test_click_right_of_apex_sets_right_bound(self):
        """A click right of the panel's peak sets right_time."""
        entries = self._two_chromatogram_entries()
        manual_bounds: dict = {}
        trace_map = {0: 0, 1: 1}
        _simulate_single_click(entries, 14.0, manual_bounds,
                               clicked_curve=1, trace_to_chrom_idx=trace_map)
        for bounds in manual_bounds.values():
            assert "right_time" in bounds
            assert bounds["right_time"] == 14.0

    def test_trace_map_with_fill_traces(self):
        """Trace map handles gaps from fill traces (e.g. trace 0→chrom 0, trace 3→chrom 1)."""
        entries = self._two_chromatogram_entries()
        manual_bounds: dict = {}
        # Simulate: chrom 0 has line at trace 0 + 2 fill traces, chrom 1 line at trace 3
        trace_map = {0: 0, 3: 1}
        _simulate_single_click(entries, 11.0, manual_bounds,
                               clicked_curve=3, trace_to_chrom_idx=trace_map)
        filenames_with_bounds = {k[0] for k in manual_bounds}
        assert "file2.csv" in filenames_with_bounds
        assert "file1.csv" not in filenames_with_bounds


class TestSeparatePanelBoundsWithCSVData:
    """Validate separate-panel click targeting with real CSV data files."""

    @staticmethod
    def _load_csv_entry(filename):
        """Load a CSV test file into a chromatogram entry dict."""
        import os
        import pandas as pd
        csv_path = os.path.join(os.path.dirname(__file__), filename)
        df = pd.read_csv(csv_path, header=None)
        df.columns = ["time", "intensity"]
        return _make_chromatogram_entry(filename, df["time"].values, df["intensity"].values)

    def test_click_on_second_csv_panel_targets_only_second_file(self):
        """Clicking the panel for TestFile2.CSV should only set bounds for that file."""
        entry1 = self._load_csv_entry("TestFile.CSV")
        entry2 = self._load_csv_entry("TestFile2.CSV")
        entries = [entry1, entry2]
        manual_bounds: dict = {}
        trace_map = {0: 0, 1: 1}
        # Click near TestFile2.CSV's first peak (~10.94)
        _simulate_single_click(entries, 10.0, manual_bounds,
                               clicked_curve=1, trace_to_chrom_idx=trace_map)
        filenames_with_bounds = {k[0] for k in manual_bounds}
        assert "TestFile2.CSV" in filenames_with_bounds
        assert "TestFile.CSV" not in filenames_with_bounds

    def test_click_on_first_csv_panel_targets_only_first_file(self):
        """Clicking the panel for TestFile.CSV should only set bounds for that file."""
        entry1 = self._load_csv_entry("TestFile.CSV")
        entry2 = self._load_csv_entry("TestFile2.CSV")
        entries = [entry1, entry2]
        manual_bounds: dict = {}
        trace_map = {0: 0, 1: 1}
        # Click near TestFile.CSV's first peak (~15.26)
        _simulate_single_click(entries, 14.5, manual_bounds,
                               clicked_curve=0, trace_to_chrom_idx=trace_map)
        filenames_with_bounds = {k[0] for k in manual_bounds}
        assert "TestFile.CSV" in filenames_with_bounds
        assert "TestFile2.CSV" not in filenames_with_bounds

    def test_independent_bounds_per_panel_from_csv(self):
        """Each panel can have its own bounds set independently."""
        entry1 = self._load_csv_entry("TestFile.CSV")
        entry2 = self._load_csv_entry("TestFile2.CSV")
        entries = [entry1, entry2]
        manual_bounds: dict = {}
        trace_map = {0: 0, 1: 1}
        # Set a left bound on the first panel near TestFile.CSV's peak at ~15.26
        _simulate_single_click(entries, 14.5, manual_bounds,
                               clicked_curve=0, trace_to_chrom_idx=trace_map)
        # Set a left bound on the second panel near TestFile2.CSV's peak at ~10.94
        _simulate_single_click(entries, 10.0, manual_bounds,
                               clicked_curve=1, trace_to_chrom_idx=trace_map)
        # Both files should have independent bounds
        filenames_with_bounds = {k[0] for k in manual_bounds}
        assert "TestFile.CSV" in filenames_with_bounds
        assert "TestFile2.CSV" in filenames_with_bounds
        assert len(manual_bounds) == 2


# ------------------------------------------------------------------
# Plot image rendering roundtrip (cached helper in app.py)
# ------------------------------------------------------------------

def _render_plot_image(fig_json: str, fmt: str, width: int, height: int) -> bytes:
    """Replicate the cached rendering helper from app.py."""
    restored_fig = go.Figure(json.loads(fig_json))
    buf = io.BytesIO()
    restored_fig.write_image(buf, format=fmt, width=width, height=height)
    return buf.getvalue()


class TestRenderPlotImage:
    """Validate the JSON-roundtrip image rendering used for cached export."""

    @staticmethod
    def _sample_figure() -> go.Figure:
        t = np.linspace(0, 20, 200)
        y = 100 * np.exp(-((t - 10) ** 2) / 8)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name="sample"))
        fig.update_layout(template="plotly_white", width=800, height=400)
        return fig

    def test_png_output_has_valid_header(self):
        fig = self._sample_figure()
        data = _render_plot_image(fig.to_json(), "png", 800, 400)
        assert data[:8] == b"\x89PNG\r\n\x1a\n"

    def test_svg_output_contains_svg_tag(self):
        fig = self._sample_figure()
        data = _render_plot_image(fig.to_json(), "svg", 800, 400)
        assert b"<svg" in data

    def test_roundtrip_preserves_trace_count(self):
        fig = self._sample_figure()
        restored = go.Figure(json.loads(fig.to_json()))
        assert len(restored.data) == len(fig.data)

    def test_roundtrip_preserves_layout_template(self):
        fig = self._sample_figure()
        restored = go.Figure(json.loads(fig.to_json()))
        assert restored.layout.template == fig.layout.template


# ------------------------------------------------------------------
# Deferred render during edit-bounds mode
# ------------------------------------------------------------------

def _should_render_plot(edit_bounds: bool) -> bool:
    """Mirror the guard in ``_plot_download_button`` in *app.py*.

    When *edit_bounds* is active the expensive Kaleido static-image
    render is skipped so that the interactive click → rerun cycle
    stays responsive.
    """
    return not edit_bounds


class TestDeferredRenderDuringEditBounds:
    """Validate that image rendering is deferred while editing bounds."""

    def test_render_skipped_when_editing_bounds(self):
        """When edit_bounds is True the expensive render must be skipped."""
        assert not _should_render_plot(edit_bounds=True)

    def test_render_active_when_not_editing_bounds(self):
        """When edit_bounds is False the render proceeds normally."""
        assert _should_render_plot(edit_bounds=False)

    def test_edit_config_does_not_alter_trace_data(self):
        """Applying edit-bounds config must not mutate the underlying data."""
        fig = _build_sample_figure()
        original_x = list(fig.data[0].x)
        original_y = list(fig.data[0].y)
        _apply_edit_bounds_config(fig)
        assert list(fig.data[0].x) == original_x
        assert list(fig.data[0].y) == original_y

    def test_chart_config_fast_without_render(self):
        """Chart configuration for edit-bounds mode should complete quickly."""
        import time
        fig = _build_sample_figure()
        start = time.perf_counter()
        _apply_edit_bounds_config(fig)
        elapsed = time.perf_counter() - start
        # Configuration should be nearly instant (<200 ms)
        assert elapsed < 0.2

    def test_render_produces_valid_output_after_config(self):
        """Render still works correctly on a figure with edit-bounds config."""
        fig = _build_sample_figure()
        _apply_edit_bounds_config(fig)
        data = _render_plot_image(fig.to_json(), "png", 800, 400)
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
