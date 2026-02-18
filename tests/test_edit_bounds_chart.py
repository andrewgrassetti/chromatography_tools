"""Tests for Plotly figure configuration when edit-bounds mode is active.

These tests validate that the chart is properly configured for interactive
peak-bound selection: crosshair cursor (dragmode="select"), single-click
support (clickmode="event+select"), and invisible markers on line traces
for reliable point-click detection.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


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


def _apply_edit_bounds_config(fig: go.Figure) -> None:
    """Apply the same configuration that app.py applies when edit_bounds is True."""
    fig.update_layout(dragmode="select", clickmode="event+select")
    for trace_data in fig.data:
        if getattr(trace_data, "mode", None) == "lines":
            trace_data.update(
                mode="lines+markers",
                marker=dict(size=8, opacity=0),
            )


class TestEditBoundsChartConfig:
    def test_dragmode_is_select(self):
        """dragmode should be 'select' so the cursor is a crosshair."""
        fig = _build_sample_figure()
        _apply_edit_bounds_config(fig)
        assert fig.layout.dragmode == "select"

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
        assert fill_trace.marker.size is None or fill_trace.marker.size != 8
