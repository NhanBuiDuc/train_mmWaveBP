"""Signal processing shared by every method: filters, I/Q correction, displacement, VMD, beats.

Pure numpy / scipy. Sources are named per function; "(ours)" marks choices the papers leave open.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache

import numpy as np
from scipy import signal

EPS = 1e-12


# ------------------------------------------------------------------ filters
@lru_cache(maxsize=64)
def _butter(order: int, cutoff: tuple[float, ...] | float, btype: str, fs: float) -> np.ndarray:
    """Second-order sections, cached: pipelines design the same filters over and over."""
    return signal.butter(order, cutoff, btype=btype, fs=fs, output="sos")

def bandpass(x: np.ndarray, fs: float, band: tuple[float, float], order: int = 4, axis: int = -1) -> np.ndarray:
    """Zero-phase Butterworth band-pass; the upper edge is clipped below Nyquist."""
    lo, hi = float(band[0]), float(min(band[1], 0.45 * fs))
    return signal.sosfiltfilt(_butter(order, (lo, hi), "bandpass", float(fs)), x, axis=axis)

def lowpass(x: np.ndarray, fs: float, cutoff: float, order: int = 4, axis: int = -1) -> np.ndarray:
    return signal.sosfiltfilt(_butter(order, float(min(cutoff, 0.45 * fs)), "lowpass", float(fs)), x, axis=axis)

def resample(x: np.ndarray, fs: float, fs_new: float, axis: int = -1) -> np.ndarray:
    """Polyphase resampling to fs_new (rational approximation of the ratio)."""
    if abs(fs - fs_new) < 1e-9:
        return np.asarray(x)
    ratio = Fraction(fs_new / fs).limit_denominator(1000)
    return signal.resample_poly(x, ratio.numerator, ratio.denominator, axis=axis)

def derivative(x: np.ndarray, fs: float, order: int = 1, window_s: float = 0.025) -> np.ndarray:
    """Savitzky-Golay derivative (noise-robust; used for SCG-like 2nd derivatives)."""
    n = max(int(round(window_s * fs)) | 1, 2 * order + 3)          # odd, longer than polyorder
    return signal.savgol_filter(x, n, max(order + 1, 3), deriv=order, delta=1 / fs)

def polynomial_detrend(x: np.ndarray, degree: int = 4) -> np.ndarray:
    t = np.linspace(-1, 1, len(x))
    return x - np.polyval(np.polyfit(t, x, degree), t)

def interpolate_peak(y: np.ndarray, i: int) -> float:
    """Sub-sample position of the extremum at index i: vertex of the parabola through
    y[i-1], y[i], y[i+1] (use log power for a spectral peak). Returns i at the edges or when the
    three points are collinear."""
    if 0 < i < len(y) - 1:
        a, b, c = float(y[i - 1]), float(y[i]), float(y[i + 1])
        den = a - 2 * b + c
        if abs(den) > 1e-12:
            return i + float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    return float(i)


# ------------------------------------------------------------------ I/Q -> displacement
def circle_fit(z: np.ndarray) -> tuple[complex, float]:
    """Taubin algebraic circle fit of I/Q samples (Vysotskaya 2023, Singh 2023 DC removal).
    Returns (centre, radius). Newton iteration after Chernov's formulation."""
    xm, ym = z.real.mean(), z.imag.mean()
    u, v = z.real - xm, z.imag - ym
    zz = u * u + v * v
    mxx, myy, mxy = (u * u).mean(), (v * v).mean(), (u * v).mean()
    mxz, myz, mzz = (u * zz).mean(), (v * zz).mean(), (zz * zz).mean()
    mz = mxx + myy
    cov_xy, var_z = mxx * myy - mxy * mxy, mzz - mz * mz
    a3, a2 = 4 * mz, -3 * mz * mz - mzz
    a1 = var_z * mz + 4 * cov_xy * mz - mxz * mxz - myz * myz
    a0 = mxz * (mxz * myy - myz * mxy) + myz * (myz * mxx - mxz * mxy) - var_z * cov_xy
    x, y = 0.0, np.inf
    for _ in range(100):
        y_old, y = y, a0 + x * (a1 + x * (a2 + x * a3))
        if abs(y) > abs(y_old):
            x = 0.0
            break
        x_new = x - y / (a1 + x * (2 * a2 + 3 * a3 * x))
        if abs(x_new - x) <= 1e-12 * max(abs(x_new), 1e-30):
            x = x_new
            break
        x = x_new
    det = x * x - x * mz + cov_xy
    cx = (mxz * (myy - x) - myz * mxy) / det / 2
    cy = (myz * (mxx - x) - mxz * mxy) / det / 2
    return complex(cx + xm, cy + ym), float(np.sqrt(cx * cx + cy * cy + mz))

def ellipse_to_circle(z: np.ndarray, fit_points: int = 200_000):
    """I/Q imbalance correction for CW / six-port radars (Erlangen: axis ratio 1.14-1.25).

    Direct least-squares ellipse fit (Fitzgibbon 1999, numerically stable form of Halir &
    Flusser 1998): A x^2 + B xy + C y^2 + D x + E y + F = 0 with 4AC - B^2 > 0. With centre c and
    Q = [[A, B/2], [B/2, C]], w = Q^(1/2) (u - c) lies on a circle, so arg(w) is the demodulated
    phase up to a constant. Returns (w as complex around 0, centre, "ellipse" | "circle").

    An ellipse has 5 degrees of freedom: on a short arc (a few mm of chest motion at 24 GHz) it
    fits noise and amplitude drift. It is kept only if it makes the radius more constant than
    the 3-parameter circle fit does; else the circle result is returned. On Erlangen resting
    records that is the case only for long arcs (~5 rad). Our FMCW radar samples a range-FFT
    bin, which has no I/Q imbalance: `circle_fit` is enough there.
    """
    s = z[:: max(1, len(z) // fit_points)]
    x, y = s.real - s.real.mean(), s.imag - s.imag.mean()
    scale = np.sqrt(np.mean(x * x + y * y)) + 1e-30          # conditioning
    x, y = x / scale, y / scale
    D1 = np.column_stack([x * x, x * y, y * y])
    D2 = np.column_stack([x, y, np.ones_like(x)])
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    T = -np.linalg.solve(S3, S2.T)
    M = S1 + S2 @ T
    M = np.array([M[2] / 2, -M[1], M[0] / 2])                 # inv(C1) @ M, C1 = [[0,0,2],[0,-1,0],[2,0,0]]
    _, vecs = np.linalg.eig(M)
    vecs = np.real(vecs)
    ok = 4 * vecs[0] * vecs[2] - vecs[1] ** 2 > 0
    c_centre, _ = circle_fit(s)
    spread = lambda v: float(np.std(np.abs(v)) / (np.mean(np.abs(v)) + 1e-30))
    circle = (z - c_centre, c_centre, "circle")
    if not ok.any():                                          # degenerate conic: no ellipse
        return circle
    a = vecs[:, np.flatnonzero(ok)[0]]
    A, B, C = a
    Dd, E, _ = T @ a
    den = B * B - 4 * A * C
    cx, cy = (2 * C * Dd - B * E) / den, (2 * A * E - B * Dd) / den
    Q = np.array([[A, B / 2], [B / 2, C]]) * np.sign(A)
    ev, V = np.linalg.eigh(Q)
    root = V @ np.diag(np.sqrt(np.clip(ev, 1e-30, None))) @ V.T
    centre = complex(cx * scale + s.real.mean(), cy * scale + s.imag.mean())
    rot = lambda v: (lambda m: m[0] + 1j * m[1])(root @ np.vstack([v.real - centre.real, v.imag - centre.imag]))
    if spread(rot(s)) >= spread(s - c_centre):
        return circle
    return rot(z), centre, "ellipse"

def displacement_mm(z: np.ndarray, wavelength_m: float, centre: complex = 0j) -> np.ndarray:
    """Unwrapped phase around `centre` -> radial displacement, d = phase * lambda / (4 pi)."""
    return np.unwrap(np.angle(z - centre)) * wavelength_m / (4 * np.pi) * 1e3


# ------------------------------------------------------------------ periodicity / quality
def autocorrelation(x: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation for lags >= 0 (FFT based)."""
    x = x - x.mean()
    n = 1 << int(np.ceil(np.log2(2 * len(x))))
    f = np.fft.rfft(x, n)
    acf = np.fft.irfft(f * np.conj(f), n)[: len(x)]
    return acf / (acf[0] + EPS)

def periodicity(x: np.ndarray, fs: float, rate_band_hz: tuple[float, float] = (0.7, 3.0)) -> tuple[float, float]:
    """Highest autocorrelation peak at a lag inside the heart-rate band -> (score, lag s).

    PolyPulse ranks bins this way (0.7-3 Hz); airBP selects the range/angle cell with the
    maximum autocorrelation instead of the maximum power.
    """
    acf = autocorrelation(x)
    lo, hi = int(fs / rate_band_hz[1]), min(len(acf) - 1, int(np.ceil(fs / rate_band_hz[0])))
    peaks, _ = signal.find_peaks(acf[: hi + 1])
    peaks = peaks[peaks >= lo]
    if not len(peaks):
        return 0.0, float("nan")
    best = peaks[np.argmax(acf[peaks])]
    return float(acf[best]), float(best / fs)

def stationarity(x: np.ndarray, fs: float, lag_s: tuple[float, float] = (2.0, 10.0)) -> float:
    """RF-BP gate: maximum normalised autocorrelation over breathing-period lags (ours: 2-10 s).
    The paper keeps a 20 s segment when this exceeds 0.6."""
    acf = autocorrelation(x)
    lo, hi = int(lag_s[0] * fs), min(len(acf), int(lag_s[1] * fs))
    return float(acf[lo:hi].max()) if hi > lo else 0.0

def dominant_rate_hz(x: np.ndarray, fs: float, band: tuple[float, float] = (0.8, 3.0)) -> float:
    """Frequency of the largest spectral peak in `band` (zero-padded Hann FFT)."""
    n = max(8192, 1 << int(np.ceil(np.log2(4 * len(x)))))
    spectrum = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x)), n))
    freqs = np.fft.rfftfreq(n, 1 / fs)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    return float(freqs[in_band][np.argmax(spectrum[in_band])])


# ------------------------------------------------------------------ decomposition
def vmd(x: np.ndarray, fs: float, k: int = 7, alpha: float = 2000.0, tol: float = 1e-7,
        max_iter: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Variational mode decomposition (Dragomiretskiy & Zosso 2014), tau = 0, no DC mode.

    Returns (modes [k, n], centre frequencies in Hz), sorted by centre frequency. TRCCBP keeps
    the modes from the heartbeat mode downwards in energy; airBP keeps modes 2 and 3.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    mirrored = np.concatenate([x[: n // 2][::-1], x, x[n // 2 + n % 2:][::-1]])
    total = len(mirrored)
    freqs = np.arange(total) / total - 0.5
    spectrum = np.fft.fftshift(np.fft.fft(mirrored))
    spectrum[: total // 2] = 0                                   # analytic signal
    u = np.zeros((k, total), dtype=complex)
    omega = 0.5 / k * np.arange(k)                               # uniform initialisation
    for _ in range(max_iter):
        u_prev = u.copy()
        for i in range(k):
            residual = spectrum - u.sum(axis=0) + u[i]
            u[i] = residual / (1 + 2 * alpha * (freqs - omega[i]) ** 2)
            half = u[i, total // 2:]
            omega[i] = (freqs[total // 2:] @ np.abs(half) ** 2) / (np.sum(np.abs(half) ** 2) + EPS)
        change = np.sum(np.abs(u - u_prev) ** 2) / (np.sum(np.abs(u_prev) ** 2) + EPS)
        if change < tol:
            break
    full = np.zeros_like(u)
    full[:, total // 2:] = u[:, total // 2:]
    full[:, 1: total // 2] = np.conj(u[:, total - 1: total // 2: -1])
    full[:, 0] = np.conj(full[:, -1])
    modes = np.real(np.fft.ifft(np.fft.ifftshift(full, axes=-1), axis=-1))[:, n // 2: n // 2 + n]
    order = np.argsort(omega)
    return modes[order], omega[order] * fs


# ------------------------------------------------------------------ beats
def ampd(x: np.ndarray, max_scale: int | None = None) -> np.ndarray:
    """Automatic multiscale-based peak detection. Returns peak indices.

    Local-maximum scalogram over scales 1..L, the scale with most maxima sets how many scales a
    point must be a maximum at. max_scale bounds memory/time (ours: 1.5 s of samples).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    scales = min(max_scale or n // 2 - 1, n // 2 - 1)
    if scales < 1:
        return np.zeros(0, dtype=int)
    is_max = np.zeros((scales, n), dtype=bool)
    for k in range(1, scales + 1):
        is_max[k - 1, k:n - k] = (x[k:n - k] > x[:n - 2 * k]) & (x[k:n - k] > x[2 * k:])
    best = int(np.argmax(is_max.sum(axis=1))) + 1
    return np.flatnonzero(is_max[:best].all(axis=0))

def detect_peaks(x: np.ndarray, fs: float, max_rate_hz: float = 3.0, prominence: float = 0.3) -> np.ndarray:
    """Systolic peaks: find_peaks with a refractory period and a robust prominence (ours)."""
    scale = np.median(np.abs(x - np.median(x))) * 1.4826 + EPS
    peaks, _ = signal.find_peaks(x, distance=max(1, int(fs / max_rate_hz)), prominence=prominence * scale)
    return peaks

def feet_before(x: np.ndarray, peaks: np.ndarray, fs: float, max_back_s: float = 0.5) -> np.ndarray:
    """Pulse foot = minimum in the window before each peak (bounded by the previous peak)."""
    feet = []
    for i, p in enumerate(peaks):
        lo = max(peaks[i - 1] if i else 0, p - int(max_back_s * fs))
        feet.append(lo + int(np.argmin(x[lo:p + 1])))
    return np.array(feet, dtype=int)

def fix_polarity(x: np.ndarray, fs: float) -> tuple[np.ndarray, bool]:
    """Flip the pulse so the upstroke (foot -> peak) is shorter than the downstroke (ours rule
    for Bahmani 2025's inversion check). Returns (signal, flipped)."""
    def rise_fraction(y):
        peaks = detect_peaks(y, fs)
        if len(peaks) < 3:
            return 0.5
        feet = feet_before(y, peaks, fs)
        period = np.median(np.diff(peaks))
        return float(np.median((peaks - feet) / period))
    flipped = rise_fraction(-x) < rise_fraction(x)
    return (-x if flipped else x), bool(flipped)

def segment(x: np.ndarray, boundaries: np.ndarray, min_len: int = 3) -> list[np.ndarray]:
    """Beats between consecutive boundary indices (e.g. feet)."""
    return [x[a:b] for a, b in zip(boundaries[:-1], boundaries[1:]) if b - a >= min_len]

def normalise(beat: np.ndarray, length: int = 100) -> np.ndarray:
    """Resample a beat to a fixed length and scale it to [0, 1]."""
    t = np.linspace(0, 1, len(beat))
    y = np.interp(np.linspace(0, 1, length), t, beat)
    return (y - y.min()) / (np.ptp(y) + EPS)

def template_correlation(beats: list[np.ndarray], template: np.ndarray | None = None,
                         length: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Pearson r of every normalised beat with a template (median beat if none given)."""
    if not beats:
        return np.zeros(0), np.zeros(length)
    stack = np.array([normalise(b, length) for b in beats])
    template = np.median(stack, axis=0) if template is None else template
    r = np.array([np.corrcoef(b, template)[0, 1] for b in stack])
    return np.nan_to_num(r), template

def interval_ok(peaks: np.ndarray, samples_per_beat: float, tolerance: float = 0.25) -> np.ndarray:
    """RF-BP aliased-peak rejection: the spacing to the next peak must match the pulse period
    from the spectrum (tolerance ours)."""
    ok = np.ones(len(peaks), dtype=bool)
    spacing = np.diff(peaks)
    ok[:-1] &= np.abs(spacing - samples_per_beat) <= tolerance * samples_per_beat
    ok[1:] &= np.abs(spacing - samples_per_beat) <= tolerance * samples_per_beat
    return ok

def pca_similarity(x: np.ndarray, peaks: np.ndarray, half_width: int) -> np.ndarray:
    """RF-BP deformed-peak rejection: cosine similarity of the window around each peak with the
    first principal component of all windows (sign aligned to the mean). NaN at the edges."""
    sim = np.full(len(peaks), np.nan)
    idx = [i for i, p in enumerate(peaks) if p - half_width >= 0 and p + half_width < len(x)]
    if len(idx) < 2:
        return sim
    windows = np.array([x[peaks[i] - half_width: peaks[i] + half_width] for i in idx])
    _, _, vt = np.linalg.svd(windows, full_matrices=False)
    pc = vt[0] if vt[0] @ windows.mean(axis=0) >= 0 else -vt[0]
    sim[idx] = windows @ pc / (np.linalg.norm(windows, axis=1) * np.linalg.norm(pc) + EPS)
    return sim
