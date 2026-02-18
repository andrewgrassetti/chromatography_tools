"""Streamlit application for interactive chromatogram peak analysis.

Upload one or more CSV files containing ``(time, intensity)`` columns and the
app performs smoothing, peak detection, trapezoidal integration, and
relative-area calculation.  Supports multiple chromatography techniques
(HPLC, GC, SEC/GPC, Ion Chromatography) through a proper OOP class hierarchy.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from chromatography.core import (
    BaseChromatogram,
    GCChromatogram,
    HPLCChromatogram,
    IonChromatogram,
    SECChromatogram,
)

# ------------------------------------------------------------------
# Technique → class mapping
# ------------------------------------------------------------------

TECHNIQUE_MAP: dict[str, type[BaseChromatogram]] = {
    "HPLC": HPLCChromatogram,
    "GC": GCChromatogram,
    "SEC/GPC": SECChromatogram,
    "Ion Chromatography": IonChromatogram,
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def read_two_col_csv(uploaded_file: Any) -> pd.DataFrame | None:
    """Read a 2-column CSV into a ``time`` / ``intensity`` DataFrame."""
    try:
        df = pd.read_csv(uploaded_file, header=None)
    except Exception:
        return None
    if df.shape[1] < 2:
        return None
    df = df.iloc[:, :2]
    df.columns = ["time", "intensity"]
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    if df.empty:
        return None
    return df


VIRIDIS_COLORS = [
    "#440154", "#482878", "#3e4989", "#31688e", "#26828e",
    "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725",
]


def _color_palette(n: int) -> list[str]:
    if n <= len(VIRIDIS_COLORS):
        step = max(1, len(VIRIDIS_COLORS) // n)
        return [VIRIDIS_COLORS[i * step % len(VIRIDIS_COLORS)] for i in range(n)]
    return [VIRIDIS_COLORS[i % len(VIRIDIS_COLORS)] for i in range(n)]


def _demo_data(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 800
    time = np.linspace(0, 20, n)
    peak1 = (80 + 10 * seed) * np.exp(-((time - (9.5 + 0.2 * seed)) ** 2) / (2 * 0.25 ** 2))
    peak2 = (60 + 10 * seed) * np.exp(-((time - (10.6 + 0.2 * seed)) ** 2) / (2 * 0.22 ** 2))
    intensity = peak1 + peak2 + rng.normal(scale=4, size=n)
    return pd.DataFrame({"time": time, "intensity": intensity})


# ------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------

st.set_page_config(page_title="Chromatography Peak Analysis", layout="wide")
st.title("Chromatography Peak Analysis")

# ---- Sidebar ----
with st.sidebar:
    st.header("Data")
    uploaded_files = st.file_uploader(
        "Upload CSV files (2 columns: time, intensity)",
        type=["csv"],
        accept_multiple_files=True,
    )
    use_demo = st.checkbox("Use demo data if no files uploaded", value=True)

    st.header("Technique")
    technique = st.selectbox(
        "Chromatography type",
        list(TECHNIQUE_MAP.keys()),
        help="Selects technique-specific defaults (axis labels, smoothing window, etc.)",
    )

    st.header("Display")
    display_mode = st.radio("Mode", ["Overlay all", "Separate panels"], horizontal=True)
    show_peaks = st.checkbox("Show peaks", value=True)
    label_peaks = st.checkbox("Label peaks", value=True)
    show_area_pct = st.checkbox("Include % area in labels", value=True)

    st.header("Peak Picking")
    min_height_pct = st.slider("Min height (% of max)", 0, 100, 10, step=1)

    st.header("Export")
    plot_fmt = st.selectbox("Save plot as", ["PNG", "SVG", "PDF"])
    plot_w = st.number_input("Width (px)", value=1000, min_value=200, step=50)
    plot_h = st.number_input("Height (px)", value=600, min_value=200, step=50)


# ---- Load data ----
datasets: list[dict[str, Any]] = []

if uploaded_files:
    for uf in uploaded_files:
        df = read_two_col_csv(uf)
        if df is not None:
            datasets.append({"filename": uf.name, "label": uf.name.rsplit(".", 1)[0], "df": df})
elif use_demo:
    for idx in (1, 2):
        datasets.append({"filename": f"demo_{chr(64 + idx)}.csv", "label": f"demo_{chr(64 + idx)}", "df": _demo_data(idx)})

if not datasets:
    st.info("Upload at least one CSV file or enable the demo data option.")
    st.stop()

# ---- Nicknames ----
with st.sidebar:
    st.header("Nicknames")
    for ds in datasets:
        ds["label"] = st.text_input(f"Nickname for {ds['filename']}", value=ds["label"], key=f"nick_{ds['filename']}")


# ---- Build Chromatogram objects (using the selected technique class) ----
ChromClass = TECHNIQUE_MAP[technique]
chromatograms: list[dict[str, Any]] = []
for ds in datasets:
    ch = ChromClass(ds["df"]["time"].values, ds["df"]["intensity"].values)
    ch.smooth()  # uses technique-specific defaults

    ref_signal = ch.smoothed if ch.smoothed is not None else ch.intensity
    min_h_abs = (min_height_pct / 100) * float(np.nanmax(ref_signal))

    ch.find_peaks(min_height=min_h_abs)
    if ch.peaks:
        ch.integrate_peaks()

    chromatograms.append({"label": ds["label"], "filename": ds["filename"], "chrom": ch})

# Axis labels from the first chromatogram instance
x_label = chromatograms[0]["chrom"].x_axis_label
y_label = chromatograms[0]["chrom"].y_axis_label

# ---- Plotting ----
colors = _color_palette(len(chromatograms))

if display_mode == "Overlay all" or len(chromatograms) == 1:
    fig = go.Figure()
    for idx, entry in enumerate(chromatograms):
        ch = entry["chrom"]
        fig.add_trace(go.Scatter(
            x=ch.time, y=ch.intensity, mode="lines",
            name=entry["label"], line=dict(color=colors[idx], width=2),
        ))
        if show_peaks and ch.peaks:
            for pk in ch.peaks:
                fig.add_vline(x=pk.time, line_dash="dot", line_color=colors[idx], opacity=0.6)
                if label_peaks:
                    txt = f"t={pk.time:.2f}"
                    if show_area_pct and pk.rel_area_pct is not None:
                        txt += f", {pk.rel_area_pct:.1f}%"
                    fig.add_annotation(x=pk.time, y=pk.height, text=txt, showarrow=False, yshift=15, font=dict(size=10))

    fig.update_layout(
        xaxis_title=x_label, yaxis_title=y_label,
        template="plotly_white", width=plot_w, height=plot_h,
        legend_title="Sample",
    )
else:
    n = len(chromatograms)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=[e["label"] for e in chromatograms])
    for idx, entry in enumerate(chromatograms):
        r, c = divmod(idx, ncols)
        ch = entry["chrom"]
        fig.add_trace(go.Scatter(
            x=ch.time, y=ch.intensity, mode="lines",
            name=entry["label"], line=dict(color=colors[idx], width=2), showlegend=False,
        ), row=r + 1, col=c + 1)
        if show_peaks and ch.peaks:
            for pk in ch.peaks:
                fig.add_vline(x=pk.time, line_dash="dot", line_color="gray", opacity=0.6, row=r + 1, col=c + 1)
                if label_peaks:
                    txt = f"t={pk.time:.2f}"
                    if show_area_pct and pk.rel_area_pct is not None:
                        txt += f", {pk.rel_area_pct:.1f}%"
                    fig.add_annotation(x=pk.time, y=pk.height, text=txt, showarrow=False, yshift=15, font=dict(size=10), xref=f"x{idx + 1}" if idx else "x", yref=f"y{idx + 1}" if idx else "y")

    fig.update_layout(template="plotly_white", width=plot_w, height=plot_h)

st.plotly_chart(fig, use_container_width=True)


# ---- Plot export ----
fmt_map = {"PNG": "png", "SVG": "svg", "PDF": "pdf"}
buf = io.BytesIO()
fig.write_image(buf, format=fmt_map[plot_fmt], width=plot_w, height=plot_h)
buf.seek(0)
st.download_button(
    label=f"Download plot ({plot_fmt})",
    data=buf,
    file_name=f"chromatograms.{fmt_map[plot_fmt]}",
    mime=f"image/{fmt_map[plot_fmt]}",
)


# ---- Summary CSV ----
rows: list[dict[str, Any]] = []
for entry in chromatograms:
    ch = entry["chrom"]
    base: dict[str, Any] = {
        "filename": entry["filename"],
        "nickname": entry["label"],
        "technique": ch.technique_name,
        "total_peaks": len(ch.peaks),
    }
    if ch.peaks:
        for i, pk in enumerate(ch.peaks, 1):
            base[f"peak_{i}_time"] = pk.time
            base[f"peak_{i}_rel_area"] = pk.rel_area_pct
    rows.append(base)

summary_df = pd.DataFrame(rows)
st.subheader("Peak Summary")
st.dataframe(summary_df, use_container_width=True)

csv_buf = io.StringIO()
summary_df.to_csv(csv_buf, index=False)
st.download_button(
    label="Download summary CSV",
    data=csv_buf.getvalue(),
    file_name="chromatogram_summary.csv",
    mime="text/csv",
)
