"""Core Chromatogram class for smoothing, peak detection, and integration."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit


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


class Chromatogram:
    """Chromatogram analysis: smoothing, Gaussian fitting, peak detection, and integration.

    Parameters
    ----------
    time : array-like
        Time axis values (e.g. minutes).
    intensity : array-like
        Signal intensity values.
    """

    def __init__(self, time: np.ndarray | list[float], intensity: np.ndarray | list[float]) -> None:
        self.time = np.asarray(time, dtype=float)
        self.intensity = np.asarray(intensity, dtype=float)
        if self.time.shape != self.intensity.shape:
            raise ValueError("time and intensity must have the same length")

        self.smoothed: np.ndarray | None = None
        self.fit_params: dict[str, float] | None = None
        self.baseline_subtracted: np.ndarray | None = None
        self.peaks: list[Peak] = []

    # ------------------------------------------------------------------
    # Smoothing
    # ------------------------------------------------------------------

    def smooth(self, window: int = 11, poly: int = 3) -> "Chromatogram":
        """Apply Savitzky-Golay smoothing.

        Parameters
        ----------
        window : int
            Window length (must be odd and <= length of data).
        poly : int
            Polynomial order.
        """
        if len(self.intensity) >= window and window % 2 == 1:
            self.smoothed = savgol_filter(self.intensity, window_length=window, polyorder=poly)
        else:
            self.smoothed = self.intensity.copy()
        return self

    # ------------------------------------------------------------------
    # Gaussian fitting
    # ------------------------------------------------------------------

    @staticmethod
    def _gaussian(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))

    def fit_gaussian(self) -> "Chromatogram":
        """Fit a single Gaussian to the smoothed signal.

        Raises
        ------
        RuntimeError
            If ``smooth()`` has not been called first.
        """
        if self.smoothed is None:
            raise RuntimeError("Smooth the signal first with .smooth()")

        p0 = [
            float(np.nanmax(self.smoothed)),
            float(self.time[np.nanargmax(self.smoothed)]),
            float((self.time[-1] - self.time[0]) / 10),
        ]
        popt, _ = curve_fit(self._gaussian, self.time, self.smoothed, p0=p0, maxfev=10_000)
        self.fit_params = {"a": popt[0], "b": popt[1], "c": popt[2]}
        return self

    # ------------------------------------------------------------------
    # Baseline subtraction
    # ------------------------------------------------------------------

    def baseline(self, method: str = "min") -> "Chromatogram":
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
        """Compute the total area under the curve using the trapezoidal rule.

        Parameters
        ----------
        use : ``"raw"`` | ``"smoothed"``
        """
        y = self.smoothed if (use == "smoothed" and self.smoothed is not None) else self.intensity
        return float(np.trapz(y, self.time))

    # ------------------------------------------------------------------
    # Peak detection
    # ------------------------------------------------------------------

    def find_peaks(self, min_height: float | None = None) -> "Chromatogram":
        """Detect peaks above *min_height* (absolute intensity).

        The algorithm mirrors the R implementation: it locates rising→falling
        transitions on the smoothed signal, resolves plateau tops, and
        determines integration bounds from surrounding local minima.

        Parameters
        ----------
        min_height : float, optional
            Minimum absolute height for a peak.  Defaults to 10 % of max.
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

        min_height_abs = min_height if min_height is not None else 0.10 * y_max

        # Identify local maxima (rising → falling transitions)
        dy = np.diff(y)
        sdy = np.sign(dy)

        # Candidate indices where slope goes from >0 to <=0
        padded_after = np.append(sdy, np.nan)
        padded_before = np.insert(sdy, 0, np.nan)
        r2f = np.where((padded_after <= 0) & (padded_before > 0))[0]

        def pick_plateau_mid(idx: int) -> int:
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
            ii = pick_plateau_mid(i) if (y[i] == y[i - 1] or y[i] == y[i + 1]) else int(i)
            if ii < 1 or ii >= n - 1:
                continue
            if np.isfinite(y[ii]) and y[ii] >= y[ii - 1] and y[ii] > y[ii + 1] and y[ii] >= min_height_abs:
                candidates.append(ii)

        candidates = sorted(set(candidates))
        if not candidates:
            self.peaks = []
            return self

        # Find surrounding local minima (falling → rising transitions)
        f2r = np.where((padded_after >= 0) & (padded_before < 0))[0]
        mins = sorted(set([0] + list(f2r) + [n - 1]))

        def left_min_for(idx: int) -> int:
            pos = np.searchsorted(mins, idx, side="right") - 1
            return mins[max(0, pos)]

        def right_min_for(idx: int) -> int:
            pos = np.searchsorted(mins, idx, side="right")
            return mins[pos] if pos < len(mins) else n - 1

        peaks: list[Peak] = []
        for c in candidates:
            start = max(0, min(left_min_for(c), n - 2))
            end = max(start + 1, min(right_min_for(c), n - 1))
            peaks.append(Peak(position=c, time=float(x[c]), height=float(y[c]), start=start, end=end))

        peaks.sort(key=lambda p: p.time)
        self.peaks = peaks
        return self

    # ------------------------------------------------------------------
    # Peak integration
    # ------------------------------------------------------------------

    def integrate_peaks(self) -> "Chromatogram":
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
            pk.area = float(np.trapz(yy, xx))

        # Keep only peaks with valid positive areas
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
            area_str = f", Area: {pk.rel_area_pct:.1f}%" if pk.rel_area_pct is not None else ""
            lines.append(f"  Peak {i}: time = {pk.time:.3f}, height = {pk.height:.1f}{area_str}")
        return "\n".join(lines)
