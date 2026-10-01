"""Recordings, public-dataset loaders, labels and subject folds.

Every loader returns `Recording`s with the pulse signal at `FS` = 250 Hz. A recording carries
either a continuous BP waveform (`abp`, mmHg, same rate) or one cuff reading (`sbp`, `dbp`).
Each method cuts its own windows from a recording (`mmwave_bp/methods/*.examples`).

Loaders
    erlangen     chest, 24 GHz CW radar, continuous BP             run on the real files
    blumio       wrist, 60 GHz wearable radar, continuous BP       written from the published file
                                                                   description, NOT run on the files (login)
    airbp        wrist, 15 public clips, cuff values               run on the real files
    npz          your own data, format in `npz` below

Download links: README.md and download_data.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from mmwave_bp import dsp

FS = 250.0
LIGHT_SPEED = 299_792_458.0
ERLANGEN_CARRIER_HZ = 24.0e9
ERLANGEN_ORDER = {"resting": 0, "valsalva": 1, "apnea": 2, "tiltup": 3, "tiltdown": 4}
FLAT_S = 0.5              # a CNAP recalibration holds the reading flat: reject windows containing one


@dataclass
class Recording:
    subject: str
    session: str
    site: str                         # "chest", "wrist" or "arm"
    pulse: np.ndarray                 # [n] one site, or [k, n] several body sites, at FS
    abp: np.ndarray | None = None     # [n] mmHg at FS (continuous reference)
    sbp: float | None = None          # cuff reading valid for the whole recording
    dbp: float | None = None
    order: int = 0                    # position within the subject's visit (0 = first, at rest)

    @property
    def n(self) -> int:
        return self.pulse.shape[-1]


# ------------------------------------------------------------------ labels
def beat_values(abp: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """SBP (maxima) and DBP (minima) per beat of a BP waveform (min distance 0.33 s). Peaks are
    found on the waveform minus its 1.5 s moving median, so a slow BP swing (Valsalva, tilt)
    does not hide beats behind a prominence set by the whole window's range."""
    beat = abp - median_filter(abp, size=max(3, int(1.5 * fs) | 1), mode="nearest")
    prom = max(5.0, 0.3 * float(np.ptp(beat)))
    peaks, _ = find_peaks(beat, distance=int(0.33 * fs), prominence=prom)
    troughs, _ = find_peaks(-beat, distance=int(0.33 * fs), prominence=prom)
    return abp[peaks], abp[troughs]


def valid_abp(abp: np.ndarray, fs: float) -> bool:
    """Reject reference artefacts: missing samples, out-of-range pressure, flat CNAP
    recalibration segments, too few beats."""
    if not np.all(np.isfinite(abp)) or abp.min() < 30 or abp.max() > 230 or np.ptp(abp) < 15:
        return False
    flat = np.r_[False, np.abs(np.diff(abp)) < 1e-3, False].astype(int)
    edges = np.flatnonzero(np.diff(flat))
    if len(edges) and np.max(edges[1::2] - edges[::2]) >= FLAT_S * fs:
        return False
    sbp, dbp = beat_values(abp, fs)
    return len(sbp) >= 4 and len(dbp) >= 4


def label(rec: Recording, start: int, stop: int) -> tuple[float, float] | None:
    """(SBP, DBP) of samples [start, stop) at FS: median beat maximum / minimum of the BP
    waveform, or the recording's cuff reading. None when the reference is unusable there."""
    if rec.abp is not None:
        abp = rec.abp[start:stop]
        if not valid_abp(abp, FS):
            return None
        sbp, dbp = beat_values(abp, FS)
        return float(np.median(sbp)), float(np.median(dbp))
    if rec.sbp is None or rec.dbp is None:
        return None
    return float(rec.sbp), float(rec.dbp)


def starts(rec: Recording, window_s: float, step_s: float) -> range:
    n, step = int(round(window_s * FS)), max(1, int(round(step_s * FS)))
    return range(0, rec.n - n + 1, step)


# ------------------------------------------------------------------ Erlangen phase 1 (chest)
def _load_mat(path: Path) -> dict:
    import scipy.io as sio
    try:
        m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        return {k: v for k, v in m.items() if not k.startswith("__")}
    except NotImplementedError:                       # MATLAB v7.3 = HDF5
        import h5py
        with h5py.File(path, "r") as f:
            return {k: np.array(f[k]).squeeze() for k in f.keys() if isinstance(f[k], h5py.Dataset)}


def _field(m: dict, *names):
    lower = {k.lower(): k for k in m}
    for n in names:
        if n.lower() in lower:
            return m[lower[n.lower()]]
    raise KeyError(f"none of {names} in {sorted(m)}")


def _erlangen_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(displacement mm, BP mmHg), both at FS. I/Q -> ellipse correction (six-port imbalance)
    -> unwrapped phase -> displacement, which does not depend on the carrier."""
    m = _load_mat(path)
    fs_radar = float(_field(m, "fs_radar"))
    z = np.asarray(_field(m, "radar_i"), float) + 1j * np.asarray(_field(m, "radar_q"), float)
    bp = np.asarray(_field(m, "tfm_bp"), float)
    fs_bp = float(_field(m, "fs_bp")) if any(k.lower() == "fs_bp" for k in m) else fs_radar * len(bp) / len(z)
    w, _, _ = dsp.ellipse_to_circle(z)
    disp = dsp.resample(dsp.displacement_mm(w, LIGHT_SPEED / ERLANGEN_CARRIER_HZ), fs_radar, FS)
    abp = dsp.resample(bp, fs_bp, FS)
    n = min(len(disp), len(abp))
    return (disp[:n] - disp[:n].mean()).astype(np.float32), abp[:n].astype(np.float32)


def erlangen(root: Path, cache: Path | None = None, log=print) -> list[Recording]:
    """Schellenberger et al., Sci Data 2020 (figshare 12186516, CC BY 4.0): 30 subjects, .mat
    files `GDNxxxx_n_Scenario.mat` anywhere under `root`. Processed arrays are cached as .npz."""
    cache = Path(cache or Path(root) / "cache_train_mmwavebp")
    cache.mkdir(parents=True, exist_ok=True)
    out = []
    for path in sorted(Path(root).rglob("GDN*.mat")):
        target = cache / f"{path.stem}_{int(FS)}hz.npz"
        if not target.exists():
            try:
                disp, abp = _erlangen_file(path)
            except Exception as exc:                                  # keep going, report
                log(f"  skip {path.name}: {exc!r}")
                continue
            np.savez_compressed(target, pulse=disp, abp=abp)
            log(f"  {path.name}: {len(disp) / FS / 60:.1f} min")
        d = np.load(target)
        parts = path.stem.split("_")
        scenario = parts[-1].lower()
        out.append(Recording(parts[0], scenario, "chest", d["pulse"], d["abp"],
                             order=ERLANGEN_ORDER.get(scenario, 9)))
    return out


# ------------------------------------------------------------------ Blumio (wrist)
def blumio(root: Path, column: int = 5, log=print) -> list[Recording]:
    """IEEE DataPort 10.21227/3yte-wz05: one CSV per subject with 6 columns (time s, BP mmHg,
    PPG, tonometer, radar transform, radar phase rad). `column` picks the radar signal (5 =
    phase, 4 = the proprietary transform). NOT run on the real files (download needs a login):
    check the column order on the first file before trusting a result."""
    out = []
    for path in sorted(Path(root).rglob("*.csv")):
        try:
            a = np.genfromtxt(path, delimiter=",", skip_header=1)
            a = a[np.all(np.isfinite(a[:, [0, 1, column]]), axis=1)]
            fs = 1.0 / float(np.median(np.diff(a[:, 0])))
            pulse, abp = dsp.resample(a[:, column], fs, FS), dsp.resample(a[:, 1], fs, FS)
        except Exception as exc:
            log(f"  skip {path.name}: {exc!r}")
            continue
        n = min(len(pulse), len(abp))
        out.append(Recording(path.stem, "rest", "wrist", (pulse[:n] - pulse[:n].mean()).astype(np.float32),
                             abp[:n].astype(np.float32)))
    return out


# ------------------------------------------------------------------ airBP public sample (wrist)
def airbp(root: Path, fs: float = 210.0, log=print) -> list[Recording]:
    """github.com/YumengLiang/dataset-of-mmwave, dataset_blood_pressure.zip (MIT): 15 clips of
    6300 samples at 210 Hz (variable `fftmax`, the amplitude of the artery cell) from 5 subjects,
    3 clips each (`test<subject>-<clip>.mat`); labels as lines `path SBP DBP HeartRate Gender` in
    groudtruth.txt. Checked on the real files."""
    root = Path(root)
    labels = {}
    for txt in root.rglob("*.txt"):
        for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            tokens = [t for t in re.split(r"[|,\s]+", line.strip()) if t]
            numbers = [float(t) for t in tokens[1:] if re.fullmatch(r"-?\d+(\.\d+)?", t)]
            if tokens and len(numbers) >= 2:
                labels[Path(tokens[0]).stem] = (numbers[0], numbers[1])
    out = []
    for path in sorted(root.rglob("*.mat")):
        if path.stem not in labels:
            log(f"  skip {path.name}: no label line")
            continue
        arrays = [np.asarray(v).squeeze() for v in _load_mat(path).values() if np.asarray(v).dtype.kind in "fiuc"]
        x = max(arrays, key=np.size)
        x = np.abs(x) if np.iscomplexobj(x) else x.astype(float)
        if x.ndim > 1:                                               # several channels: the longest axis is time
            x = np.moveaxis(x, int(np.argmax(x.shape)), -1).reshape(-1, max(x.shape)).mean(axis=0)
        pulse = dsp.resample(-(x - x.mean()), fs, FS)               # the pulse is -RSS (airBP Eq. 10)
        sbp, dbp = labels[path.stem]
        subject, _, clip = path.stem.partition("-")                  # test33-2 = subject test33, clip 2
        out.append(Recording(subject, path.stem, "wrist", pulse.astype(np.float32), sbp=sbp, dbp=dbp,
                             order=int(clip) if clip.isdigit() else 0))
    return out


# ------------------------------------------------------------------ your own data
def npz(root: Path, log=print) -> list[Recording]:
    """Any folder of .npz files with the keys
        pulse    [n] or [k, n] (k body sites, e.g. arm sites for hBP-Fi), any unit
        fs       sampling rate of `pulse` in Hz
        subject  text
      and either  abp [m] + fs_abp   (continuous BP in mmHg, same time span as `pulse`)
      or          sbp, dbp           (one cuff reading for the recording)
      optional    session (text), site ("chest" / "wrist" / "arm"), order (int, 0 = at rest)."""
    out = []
    for path in sorted(Path(root).rglob("*.npz")):
        d = np.load(path, allow_pickle=False)
        try:
            fs = float(d["fs"])
            pulse = dsp.resample(np.asarray(d["pulse"], float), fs, FS).astype(np.float32)
            rec = Recording(str(d["subject"]), str(d["session"]) if "session" in d else path.stem,
                            str(d["site"]) if "site" in d else "chest", pulse,
                            order=int(d["order"]) if "order" in d else 0)
            if "abp" in d:
                rec.abp = dsp.resample(np.asarray(d["abp"], float), float(d["fs_abp"]), FS).astype(np.float32)
                n = min(rec.n, len(rec.abp))
                rec.pulse, rec.abp = rec.pulse[..., :n], rec.abp[:n]
            else:
                rec.sbp, rec.dbp = float(d["sbp"]), float(d["dbp"])
        except Exception as exc:
            log(f"  skip {path.name}: {exc!r}")
            continue
        out.append(rec)
    return out


LOADERS = {"erlangen": erlangen, "blumio": blumio, "airbp": airbp, "npz": npz}


# ------------------------------------------------------------------ folds
def subject_folds(subjects, k: int, seed: int = 0) -> list[list[str]]:
    """k folds of whole subjects: nobody is in both the training and the test set."""
    unique = sorted(set(subjects))
    order = np.random.default_rng(seed).permutation(len(unique))
    return [[unique[i] for i in order[f::k]] for f in range(min(k, len(unique)))]
