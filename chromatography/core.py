"""Core chromatogram classes for smoothing, peak detection, and integration.

Provides an OOP hierarchy supporting different chromatography data types:

* :class:`BaseChromatogram` — abstract base with shared signal-processing logic.
* :class:`HPLCChromatogram` — HPLC / UV-Vis absorbance data.
* :class:`GCChromatogram` — Gas chromatography (FID / TCD).
* :class:`SECChromatogram` — Size-exclusion / GPC with optional MW calibration.
* :class:`IonChromatogram` — Ion chromatography (conductivity detection).

The convenience alias :class:`Chromatogram` maps to :class:`HPLCChromatogram`
for backward compatibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class Peak:
    """A single detected chromatographic peak."""

    position: int
    time: float
    height: float
    start: int
    end: int
    area: float | None = None
    rel_area_pct: float | None = None


# ======================================================================
# Abstract base
# ======================================================================

class BaseChromatogram(ABC):
    """Abstract base class for all chromatogram types.

    Subclasses **must** implement :meth:`x_axis_label`,
    :meth:`y_axis_label`, and :meth:`technique_name` so that
    downstream code (plotting, export) can adapt to the data type.

    Parameters
    ----------
    time : array-like
        Time / volume / elution axis values.
    intensity : array-like
        Detector response values.
    """

    def __init__(
        self,
        time: np.ndarray | list[float],
        intensity: np.ndarray | list[float],
    ) -> None:
        self.time = np.asarray(time, dtype=float)
        self.intensity = np.asarray(intensity, dtype=float)
        if self.time.shape != self.intensity.shape:
            raise ValueError("time and intensity must have the same length")

        self.smoothed: np.ndarray | None = None
        self.fit_params: dict[str, float] | None = None
        self.baseline_subtracted: np.ndarray | None = None
        self.peaks: list[Peak] = []

    # ------------------------------------------------------------------
    # Abstract properties — subclasses define these
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def technique_name(self) -> str:
        """Human-readable name of the chromatographic technique."""

    @property
    @abstractmethod
    def x_axis_label(self) -> str:
        """Label for the x (time / volume) axis."""

    @property
    @abstractmethod
    def y_axis_label(self) -> str:
        """Label for the y (response) axis."""

    @property
    def default_smooth_window(self) -> int:
        """Default Savitzky-Golay window length (overridable)."""
        return 11

    @property
    def default_smooth_poly(self) -> int:
        """Default Savitzky-Golay polynomial order (overridable)."""
        return 3

    @property
    def default_min_height_frac(self) -> float:
        """Default minimum peak-height as a fraction of max signal."""
        return 0.10

    # ------------------------------------------------------------------
    # Smoothing
    # ------------------------------------------------------------------

    def smooth(
        self,
        window: int | None = None,
        poly: int | None = None,
    ) -> "BaseChromatogram":
        """Apply Savitzky-Golay smoothing.

        Parameters
        ----------
        window : int, optional
            Window length (must be odd and ≤ data length).
            Defaults to :attr:`default_smooth_window`.
        poly : int, optional
            Polynomial order.  Defaults to :attr:`default_smooth_poly`.
        """
        window = window if window is not None else self.default_smooth_window
        poly = poly if poly is not None else self.default_smooth_poly
        if len(self.intensity) >= window and window % 2 == 1:
            self.smoothed = savgol_filter(
                self.intensity, window_length=window, polyorder=poly,
            )
        else:
            self.smoothed = self.intensity.copy()
        return self

    # ------------------------------------------------------------------
    # Gaussian fitting
    # ------------------------------------------------------------------

    @staticmethod
    def _gaussian(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))

    def fit_gaussian(self) -> "BaseChromatogram":
        """Fit a single Gaussian to the smoothed signal.

        Raises
        ------
        RuntimeError
            If :meth:`smooth` has not been called first.
        """
        if self.smoothed is None:
            raise RuntimeError("Smooth the signal first with .smooth()")

        p0 = [
            float(np.nanmax(self.smoothed)),
            float(self.time[np.nanargmax(self.smoothed)]),
            float((self.time[-1] - self.time[0]) / 10),
        ]
        popt, _ = curve_fit(
            self._gaussian, self.time, self.smoothed, p0=p0, maxfev=10_000,
        )
        self.fit_params = {"a": popt[0], "b": popt[1], "c": popt[2]}
        return self

    # ------------------------------------------------------------------
    # Baseline subtraction
    # ------------------------------------------------------------------

    def baseline(self, method: str = "min") -> "BaseChromatogram":
        """Subtract a simple baseline.

        Parameters
        ----------
        method : ``"min"`` | ``"median"``
        """
        if method == "min":
            base = float(np.nanmin(self.intensity))
        elif method == "median":
            base = float(np.nanmedian(self.intensity))
        else:
            raise ValueError(f"Unknown baseline method: {method!r}")
        self.baseline_subtracted = self.intensity - base
        return self

    # ------------------------------------------------------------------
    # Area under curve
    # ------------------------------------------------------------------

    def auc(self, use: str = "raw") -> float:
        """Total area under the curve (trapezoidal rule).

        Parameters
        ----------
        use : ``"raw"`` | ``"smoothed"``
        """
        y = (
            self.smoothed
            if (use == "smoothed" and self.smoothed is not None)
            else self.intensity
        )
        return float(np.trapezoid(y, self.time))

    # ------------------------------------------------------------------
    # Peak detection
    # ------------------------------------------------------------------

    def find_peaks(self, min_height: float | None = None) -> "BaseChromatogram":
        """Detect peaks above *min_height* (absolute intensity).

        The algorithm locates rising→falling transitions on the smoothed
        signal, resolves plateau tops, and determines integration bounds
        from surrounding local minima.

        Parameters
        ----------
        min_height : float, optional
            Minimum absolute height.  Defaults to
            ``default_min_height_frac * max(signal)``.
        """
        y = self.smoothed if self.smoothed is not None else self.intensity
        y = np.asarray(y, dtype=float)
        x = np.asarray(self.time, dtype=float)
        n = len(y)

        if n < 3:
            self.peaks = []
            return self

        y_max = np.nanmax(y)
        if not np.isfinite(y_max):
            self.peaks = []
            return self

        min_height_abs = (
            min_height if min_height is not None
            else self.default_min_height_frac * y_max
        )

        dy = np.diff(y)
        sdy = np.sign(dy)

        padded_after = np.append(sdy, np.nan)
        padded_before = np.insert(sdy, 0, np.nan)
        r2f = np.where((padded_after <= 0) & (padded_before > 0))[0]

        def _pick_plateau_mid(idx: int) -> int:
            left = right = idx
            while left > 0 and y[left - 1] == y[idx]:
                left -= 1
            while right < n - 1 and y[right + 1] == y[idx]:
                right += 1
            return int(round((left + right) / 2))

        candidates: list[int] = []
        for i in r2f:
            if i < 1 or i >= n - 1:
                continue
            ii = (
                _pick_plateau_mid(i)
                if (y[i] == y[i - 1] or y[i] == y[i + 1])
                else int(i)
            )
            if ii < 1 or ii >= n - 1:
                continue
            if (
                np.isfinite(y[ii])
                and y[ii] >= y[ii - 1]
                and y[ii] > y[ii + 1]
                and y[ii] >= min_height_abs
            ):
                candidates.append(ii)

        candidates = sorted(set(candidates))
        if not candidates:
            self.peaks = []
            return self

        f2r = np.where((padded_after >= 0) & (padded_before < 0))[0]
        mins = sorted(set([0] + list(f2r) + [n - 1]))

        def _left_min(idx: int) -> int:
            pos = np.searchsorted(mins, idx, side="right") - 1
            return mins[max(0, pos)]

        def _right_min(idx: int) -> int:
            pos = np.searchsorted(mins, idx, side="right")
            return mins[pos] if pos < len(mins) else n - 1

        peaks: list[Peak] = []
        for c in candidates:
            start = max(0, min(_left_min(c), n - 2))
            end = max(start + 1, min(_right_min(c), n - 1))
            peaks.append(
                Peak(
                    position=c,
                    time=float(x[c]),
                    height=float(y[c]),
                    start=start,
                    end=end,
                ),
            )

        peaks.sort(key=lambda p: p.time)
        self.peaks = peaks
        return self

    # ------------------------------------------------------------------
    # Peak integration
    # ------------------------------------------------------------------

    def integrate_peaks(self) -> "BaseChromatogram":
        """Compute trapezoidal areas and relative percentages for detected peaks.

        Raises
        ------
        RuntimeError
            If no peaks have been detected yet.
        """
        if not self.peaks:
            raise RuntimeError("No peaks detected. Run .find_peaks() first.")

        for pk in self.peaks:
            s = max(0, min(pk.start, len(self.time) - 1))
            e = max(0, min(pk.end, len(self.time) - 1))
            if e - s + 1 < 2:
                pk.area = None
                continue
            xx = self.time[s : e + 1]
            yy = self.intensity[s : e + 1]
            if len(xx) < 2 or np.any(np.isnan(xx)) or np.any(np.isnan(yy)):
                pk.area = None
                continue
            pk.area = float(np.trapezoid(yy, xx))

        self.peaks = [p for p in self.peaks if p.area is not None and p.area > 0]

        total = sum(p.area for p in self.peaks)  # type: ignore[arg-type]
        if total > 0:
            for p in self.peaks:
                p.rel_area_pct = (p.area / total) * 100  # type: ignore[operator]

        return self

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summarize_peaks(self) -> str:
        """Return a human-readable summary of detected peaks."""
        if not self.peaks:
            return "No peaks detected."

        lines = [f"Detected {len(self.peaks)} peak(s):"]
        for i, pk in enumerate(self.peaks, 1):
            area_str = (
                f", Area: {pk.rel_area_pct:.1f}%"
                if pk.rel_area_pct is not None
                else ""
            )
            lines.append(
                f"  Peak {i}: time = {pk.time:.3f}, "
                f"height = {pk.height:.1f}{area_str}",
            )
        return "\n".join(lines)


# ======================================================================
# Concrete chromatography types
# ======================================================================

class HPLCChromatogram(BaseChromatogram):
    """High-Performance Liquid Chromatography (UV/Vis absorbance).

    Typical data: retention time (min) vs absorbance (AU).
    """

    @property
    def technique_name(self) -> str:
        return "HPLC"

    @property
    def x_axis_label(self) -> str:
        return "Retention Time (min)"

    @property
    def y_axis_label(self) -> str:
        return "Absorbance (AU)"


class GCChromatogram(BaseChromatogram):
    """Gas Chromatography (FID / TCD detection).

    GC peaks are typically sharper than HPLC peaks, so a narrower default
    smoothing window is used.
    """

    @property
    def technique_name(self) -> str:
        return "GC"

    @property
    def x_axis_label(self) -> str:
        return "Retention Time (min)"

    @property
    def y_axis_label(self) -> str:
        return "Detector Response"

    @property
    def default_smooth_window(self) -> int:
        return 7

    @property
    def default_min_height_frac(self) -> float:
        return 0.05


class SECChromatogram(BaseChromatogram):
    """Size-Exclusion Chromatography / GPC.

    Extends the base class with optional molecular-weight calibration so
    that elution volume can be mapped to log(MW).

    Parameters
    ----------
    time : array-like
        Elution volume (mL) or time (min).
    intensity : array-like
        Detector response (refractive index, UV, etc.).
    mw_calibration : callable, optional
        A function ``f(volume) -> molecular_weight`` that converts
        elution volume to molecular weight.  If provided,
        :meth:`molecular_weights` returns the calibrated values.
    """

    def __init__(
        self,
        time: np.ndarray | list[float],
        intensity: np.ndarray | list[float],
        mw_calibration: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        super().__init__(time, intensity)
        self._mw_calibration = mw_calibration

    @property
    def technique_name(self) -> str:
        return "SEC/GPC"

    @property
    def x_axis_label(self) -> str:
        return "Elution Volume (mL)"

    @property
    def y_axis_label(self) -> str:
        return "RI Response"

    @property
    def default_smooth_window(self) -> int:
        return 15

    @property
    def default_min_height_frac(self) -> float:
        return 0.05

    def molecular_weights(self) -> np.ndarray | None:
        """Return molecular weights from the calibration curve, if available."""
        if self._mw_calibration is None:
            return None
        return self._mw_calibration(self.time)


class IonChromatogram(BaseChromatogram):
    """Ion Chromatography (conductivity detection).

    Ion chromatography peaks tend to be broader and may ride on a rising
    baseline, so a wider smoothing window is the default.
    """

    @property
    def technique_name(self) -> str:
        return "IC"

    @property
    def x_axis_label(self) -> str:
        return "Retention Time (min)"

    @property
    def y_axis_label(self) -> str:
        return "Conductivity (µS/cm)"

    @property
    def default_smooth_window(self) -> int:
        return 15

    @property
    def default_min_height_frac(self) -> float:
        return 0.08


# ======================================================================
# Backward-compatible alias
# ======================================================================

Chromatogram = HPLCChromatogram
"""Default chromatogram type (alias for :class:`HPLCChromatogram`)."""
