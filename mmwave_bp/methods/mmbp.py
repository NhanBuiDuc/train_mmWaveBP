"""mmBP (Shi et al., SenSys 2022): wrist pulse -> motion compensation -> beat features -> forest.

Paper: TI IWR1843BOOST + DCA1000 (77 GHz), wrist 5 cm under the radar, 25 subjects, 25 s
samples, arm-cuff reference, subject-level leave-one-out, no calibration.
Pipeline: phase -> delay-Doppler feature transform (DDFT) -> TR-FLAF motion compensation ->
per-pulse features MP, FIP, MMR, MIR, PPI, mean, variance -> random forest (500 trees, depth 3).

What is and is not reproduced
* DDFT: NOT implemented. The paper gives neither the window, the grid sizes nor the rule that
  picks the Doppler row, so it cannot be rebuilt; a 0.7-8 Hz band-pass stands in for it.
* TR-FLAF: implemented as an interpretation. The paper cuts 500 ms segments (which it says span
  1-2 pulse periods: only true above 120 bpm) and does not say how the one reference segment
  runs along the signal. Here the segments are the beats themselves; the reference is the beat
  that correlates above 0.86 with the most other beats; it is replayed at every detected beat
  and a trigonometric functional-link adaptive filter maps it onto the measured signal.
  Eq. 13 is printed without the error term; the normalised LMS update is used.
  Q = 2, K = 8 taps, step 0.5 are ours (not given).
* Features: MMR = max / min is undefined for a zero-mean signal; here (max - mean) / (mean - min)
  measured from the beat minimum. The inflection rule is not given; first - -> + turn of the
  2nd derivative after the peak (smoothed).

scikit-learn is needed only for this method (`pip install scikit-learn`).
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from mmwave_bp import data, dsp

TARGET = "values"
KIND = "sklearn"
FS = 200.0
WINDOW_S = 25.0
TRAIN_STEP_S = 5.0
TEST_STEP_S = 25.0
FEATURES = ("mp", "fip", "mmr", "mir", "ppi_s", "mean", "var")
EPS = 1e-12


# ------------------------------------------------------------------ TR-FLAF
def reference_beat(x: np.ndarray, peaks: np.ndarray, length: int = 100, threshold: float = 0.86):
    """TRSE: the beat (peak to peak, resampled to `length`) whose correlation with the other
    beats exceeds `threshold` most often. Returns (reference, count) or (None, 0)."""
    beats = [x[a:b] for a, b in zip(peaks[:-1], peaks[1:]) if b - a >= 3]
    if len(beats) < 3:
        return None, 0
    stack = np.array([np.interp(np.linspace(0, 1, length), np.linspace(0, 1, len(b)), b) for b in beats])
    gamma = np.nan_to_num(np.corrcoef(stack))                         # Eq. 7
    alpha = (gamma > threshold).sum(axis=1)                          # Eq. 8
    best = int(np.argmax(alpha))
    return stack[best], int(alpha[best])


def flaf(reference: np.ndarray, measured: np.ndarray, q: int = 2, taps: int = 8, step: float = 0.5) -> np.ndarray:
    """Trigonometric functional-link adaptive filter (Eq. 9-13 with the NLMS update):
    v = [c, sin(pi c), cos(pi c), ..., sin(q pi c), cos(q pi c)] over `taps` delayed samples of
    the reference c; output w^T v follows `measured` where the reference explains it."""
    c = reference / (np.max(np.abs(reference)) + EPS)
    padded = np.r_[np.zeros(taps - 1), c]
    lagged = np.lib.stride_tricks.sliding_window_view(padded, taps)   # [n, taps]
    v = np.concatenate([lagged] + [f(k * np.pi * lagged) for k in range(1, q + 1) for f in (np.sin, np.cos)], axis=1)
    w, out = np.zeros(v.shape[1]), np.zeros(len(c))
    for _ in range(2):                                               # 2nd pass starts from adapted weights
        for n in range(len(c)):
            out[n] = w @ v[n]
            w += step * (measured[n] - out[n]) * v[n] / (EPS + v[n] @ v[n])
    return out


def compensate(x: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """TR-FLAF: replay the reference beat between consecutive peaks, adapt it to the signal."""
    ref, count = reference_beat(x, peaks)
    if ref is None or count < 2:
        return x
    tiled = np.zeros_like(x)
    for a, b in zip(peaks[:-1], peaks[1:]):
        tiled[a:b] = np.interp(np.linspace(0, 1, b - a), np.linspace(0, 1, len(ref)), ref)
    out = x.copy()
    a, b = peaks[0], peaks[-1]
    out[a:b] = flaf(tiled[a:b], x[a:b])
    return out


# ------------------------------------------------------------------ features
def beat_features(beat: np.ndarray, fs: float) -> dict[str, float]:
    p = beat - beat.min()
    i_s = int(np.argmax(p))
    n = len(p)
    smooth = signal.savgol_filter(p, min(n - (n % 2 == 0), max(5, int(0.05 * fs) | 1)), 3) if n >= 7 else p
    d2 = np.gradient(np.gradient(smooth))[i_s:]
    turn = np.flatnonzero((d2[:-1] < 0) & (d2[1:] >= 0))
    fip = float(p[i_s + turn[0]]) if len(turn) else float("nan")
    mean = p.mean()
    return {"mp": float(p.max()), "fip": fip, "mmr": float((p.max() - mean) / (mean + EPS)),
            "mir": float(p.max() / (fip + EPS)) if np.isfinite(fip) else float("nan"),
            "mean": float(mean), "var": float(p.var())}


def window_features(x: np.ndarray, fs: float) -> np.ndarray | None:
    """[7] features of one window (median over its beats), None when it has too few beats."""
    x = dsp.bandpass(x, fs, (0.7, 8.0))
    x, _ = dsp.fix_polarity(x / (x.std() + EPS), fs)
    peaks = dsp.detect_peaks(x, fs)
    if len(peaks) < 5:
        return None
    x = compensate(x, peaks)
    peaks = dsp.detect_peaks(x, fs)
    feet = dsp.feet_before(x, peaks, fs)
    rows = [beat_features(b, fs) for b in dsp.segment(x, feet, min_len=int(0.3 * fs))]
    if len(rows) < 3:
        return None
    agg = {k: np.nanmedian([r[k] for r in rows]) for k in rows[0]}
    agg["ppi_s"] = float(np.median(np.diff(peaks)) / fs)
    out = np.array([agg[k] for k in FEATURES], np.float32)
    return out if np.all(np.isfinite(out)) else None


def examples(rec: data.Recording, step_s: float):
    """X [m, 7], Y [m, 2], BP [m, 2], start sample at data.FS [m]."""
    if rec.pulse.ndim != 1:
        raise ValueError("mmBP takes one pulse signal per recording")
    x = dsp.resample(rec.pulse, data.FS, FS)
    n = int(WINDOW_S * FS)
    xs, bps, where = [], [], []
    for s in range(0, len(x) - n + 1, max(1, int(step_s * FS))):
        a, b = int(s * data.FS / FS), int((s + n) * data.FS / FS)
        bp = data.label(rec, a, b)
        if bp is None:
            continue
        f = window_features(x[s: s + n], FS)
        if f is None:
            continue
        xs.append(f); bps.append(bp); where.append(a)
    bp = np.array(bps, np.float32).reshape(-1, 2)
    return np.array(xs, np.float32).reshape(-1, len(FEATURES)), bp, bp, np.array(where, int)


# ------------------------------------------------------------------ regression
def fit(x: np.ndarray, y: np.ndarray, trees: int = 500, depth: int = 3, seed: int = 0):
    """Random forest, 500 trees of depth 3, squared-error criterion (the paper's best)."""
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as exc:
        raise SystemExit("mmBP needs scikit-learn: pip install scikit-learn") from exc
    return RandomForestRegressor(n_estimators=trees, max_depth=depth, criterion="squared_error",
                                 random_state=seed, n_jobs=-1).fit(x, y)


def predict(model, x: np.ndarray) -> np.ndarray:
    return model.predict(x)
