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
from scipy.signal import find_peaks as _scipy_find_peaks, savgol_filter
from scipy.optimize import curve_fit

# np.trapezoid was added in NumPy 2.0; fall back to the older np.trapz alias.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


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
        return 0.05

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
        return float(_trapezoid(y, self.time))

    # ------------------------------------------------------------------
    # Peak detection
    # ------------------------------------------------------------------

    def find_peaks(
        self,
        min_height: float | None = None,
        prominence: float | None = None,
        distance: int | None = None,
    ) -> "BaseChromatogram":
        """Detect peaks using scipy's prominence-based algorithm.

        Uses :func:`scipy.signal.find_peaks` with prominence filtering
        for robust chromatographic peak identification.  Integration
        bounds for each peak are derived from the prominence base
        positions (left and right bases).

        Parameters
        ----------
        min_height : float, optional
            Minimum absolute height.  Defaults to
            ``default_min_height_frac * max(signal)``.
        prominence : float, optional
            Minimum peak prominence (vertical distance from the peak to
            the higher of its two neighbouring bases).  Defaults to
            ``0.01 * max(signal)``.
        distance : int, optional
            Minimum number of samples between neighbouring peaks.
            Defaults to ``max(3, len(signal) // 100)``.
        """
        y = self.smoothed if self.smoothed is not None else self.intensity
        y = np.asarray(y, dtype=float)
        x = np.asarray(self.time, dtype=float)
        n = len(y)

        if n < 3:
            self.peaks = []
            return self

        y_max = float(np.nanmax(y))
        if not np.isfinite(y_max):
            self.peaks = []
            return self

        min_height_abs = (
            min_height if min_height is not None
            else self.default_min_height_frac * y_max
        )
        prom = prominence if prominence is not None else 0.01 * y_max
        dist = distance if distance is not None else max(3, n // 100)

        indices, properties = _scipy_find_peaks(
            y, height=min_height_abs, prominence=prom, distance=dist,
        )

        if len(indices) == 0:
            self.peaks = []
            return self

        left_bases = properties["left_bases"]
        right_bases = properties["right_bases"]

        # Sort peak indices by position so we can resolve overlapping bounds.
        order = np.argsort(indices)
        sorted_indices = indices[order]
        sorted_left = left_bases[order]
        sorted_right = right_bases[order]

        # Resolve overlapping bounds between adjacent peaks by splitting
        # at the valley minimum between their apexes.
        starts = [int(max(0, min(sorted_left[k], n - 2))) for k in range(len(sorted_indices))]
        ends = [int(max(starts[k] + 1, min(sorted_right[k], n - 1))) for k in range(len(sorted_indices))]

        for k in range(len(sorted_indices) - 1):
            if ends[k] >= starts[k + 1]:
                # Find valley (minimum) between the two peak apexes
                left_apex = int(sorted_indices[k])
                right_apex = int(sorted_indices[k + 1])
                valley_idx = left_apex + int(np.argmin(y[left_apex:right_apex + 1]))
                ends[k] = valley_idx
                starts[k + 1] = valley_idx

        peaks: list[Peak] = []
        for k in range(len(sorted_indices)):
            idx = int(sorted_indices[k])
            start = starts[k]
            end = max(start + 1, ends[k])
            peaks.append(
                Peak(
                    position=idx,
                    time=float(x[idx]),
                    height=float(y[idx]),
                    start=start,
                    end=end,
                ),
            )

        peaks.sort(key=lambda p: p.time)
        self.peaks = peaks
        return self

    # ------------------------------------------------------------------
    # Manual peak-bound override
    # ------------------------------------------------------------------

    def set_manual_bounds(
        self,
        peak_time: float,
        *,
        left_time: float | None = None,
        right_time: float | None = None,
    ) -> "BaseChromatogram":
        """Override integration bounds for the peak nearest to *peak_time*.

        Only the explicitly supplied bound is changed; the other retains
        its automatically detected value.  After updating, the method
        validates the resulting bounds:

        * If the new start index ends up at or beyond the peak apex, the
          peak is removed (no valid left-side area).
        * If the new end index ends up at or before the peak apex, the
          peak is removed (no valid right-side area).

        Parameters
        ----------
        peak_time : float
            Approximate retention time of the peak whose bounds should
            be adjusted.  The closest detected peak is selected.
        left_time : float, optional
            New left (start) bound expressed as a time value.
        right_time : float, optional
            New right (end) bound expressed as a time value.

        Returns
        -------
        BaseChromatogram
            ``self``, for method chaining.

        Raises
        ------
        RuntimeError
            If no peaks have been detected yet.
        ValueError
            If neither *left_time* nor *right_time* is provided.
        """
        if not self.peaks:
            raise RuntimeError("No peaks detected. Run .find_peaks() first.")
        if left_time is None and right_time is None:
            raise ValueError(
                "At least one of left_time or right_time must be provided."
            )

        # Find the peak closest to the requested time.
        pk = min(self.peaks, key=lambda p: abs(p.time - peak_time))
        x = np.asarray(self.time, dtype=float)

        if left_time is not None:
            new_start = int(np.searchsorted(x, left_time))
            new_start = max(0, min(new_start, len(x) - 1))
            pk.start = new_start

        if right_time is not None:
            new_end = int(np.searchsorted(x, right_time))
            new_end = max(0, min(new_end, len(x) - 1))
            pk.end = new_end

        # Validate: bounds must still straddle the peak apex.
        if pk.start >= pk.position or pk.end <= pk.position:
            self.peaks = [p for p in self.peaks if p is not pk]

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
            pk.area = float(_trapezoid(yy, xx))

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
