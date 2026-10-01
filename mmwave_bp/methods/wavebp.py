"""WaveBP (Hu et al., IMWUT 2024): radar -> continuous arterial BP waveform, SBP / DBP per beat.

mmFormer: IQ conv, 4-level U-Net with (1 x 11) convs, InstanceNorm, LeakyReLU(0.3), spatially
aware attention shortcuts across N_b range bins, Transformer bottleneck, transposed-conv decoder,
5 expert regressors gated by the AHA BP category (Type I personalisation).
Loss: alpha * MSE + beta * (1 - Pearson), alpha = 1, beta = 100 (the printed correlation term is
Pearson itself; minimising it would anti-correlate, so 1 - rho is used).

Adaptations (ours): inputs are chest displacement (mm) and its 2nd derivative instead of raw I/Q
(the phase of raw I/Q scales with the carrier, so it does not transfer between radars); N_b = 1
range bin unless the data set provides more. Paper size: d_model 256, 12 layers. Not implemented:
the ECG/PPG teacher (cross-modal distillation) and BeamDA (needs multi-antenna raw data).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
import torch
from torch import nn

from mmwave_bp.dsp import bandpass, derivative, resample

# ------------------------------------------------------------------ inputs and read-out
FS = 125.0
WINDOW = 1024                                   # 8.19 s at 125 Hz


def aha_category(sbp: float, dbp: float) -> int:
    """0 hypotension, 1 normal, 2 elevated, 3 hypertension stage 1, 4 stage 2 (AHA 2017)."""
    if sbp < 90 or dbp < 60:
        return 0
    if sbp >= 140 or dbp >= 90:
        return 4
    if sbp >= 130 or dbp >= 80:
        return 3
    if sbp >= 120:
        return 2
    return 1


# ------------------------------------------------------------------ inputs
def radar_channels(displacement_mm: np.ndarray, fs: float) -> np.ndarray:
    """[2, n] at 125 Hz: 0.5-15 Hz displacement and its 2nd derivative (not yet normalised)."""
    x = resample(bandpass(displacement_mm, fs, (0.5, 15.0)), fs, FS)
    return np.stack([x, derivative(x, FS, order=2, window_s=0.06)])


def normalise_window(w: np.ndarray) -> np.ndarray:
    """Per window and channel: zero mean, unit SD (radar gain and distance drop out)."""
    return (w - w.mean(axis=-1, keepdims=True)) / (w.std(axis=-1, keepdims=True) + 1e-9)


def windows(channels: np.ndarray, step: int, abp: np.ndarray | None = None):
    """Yield (start, input [2, 1, WINDOW], abp window or None)."""
    n = channels.shape[-1]
    for s in range(0, n - WINDOW + 1, step):
        yield s, normalise_window(channels[:, s: s + WINDOW])[:, None, :].astype(np.float32), \
            (abp[s: s + WINDOW].astype(np.float32) if abp is not None else None)


EDGE = int(FS)              # 1 s at each window end: WaveBP reports the edges as least accurate


def beat_values(abp: np.ndarray, fs: float = FS) -> tuple[np.ndarray, np.ndarray]:
    """SBP (maxima) and DBP (minima) per beat of a BP waveform (min distance 0.33 s). Peaks are
    found on the waveform minus its 1.5 s moving median, so a slow BP swing (Valsalva, tilt)
    does not hide beats behind a prominence set by the whole window's range."""
    beat = abp - median_filter(abp, size=max(3, int(1.5 * fs) | 1), mode="nearest")
    prom = max(5.0, 0.3 * float(np.ptp(beat)))
    peaks, _ = find_peaks(beat, distance=int(0.33 * fs), prominence=prom)
    troughs, _ = find_peaks(-beat, distance=int(0.33 * fs), prominence=prom)
    return abp[peaks], abp[troughs]


def bp_from_waveform(abp: np.ndarray) -> tuple[float, float]:
    """(SBP, DBP) of one predicted window: median of beat maxima / minima away from the window
    edges (`EDGE`), or the 95th / 5th percentile when no beat can be segmented (a nearly flat
    prediction)."""
    core = abp[EDGE:-EDGE] if len(abp) > 4 * EDGE else abp
    sbp, dbp = beat_values(core)
    if len(sbp) and len(dbp):
        return float(np.median(sbp)), float(np.median(dbp))
    return float(np.percentile(core, 95)), float(np.percentile(core, 5))


# ------------------------------------------------------------------ model
def _block(ci: int, co: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(ci, co, (1, 11), (1, stride), (0, 5)), nn.InstanceNorm2d(co, affine=True),
                         nn.LeakyReLU(0.3))

class SpatialAttentionShortcut(nn.Module):
    """Eq. 5: per feature channel a bin-to-bin attention C = softmax(Q K^T / sqrt(D)), then an FC
    that collapses the N_b bins -> [B, D, T]."""

    def __init__(self, d: int, n_bins: int):
        super().__init__()
        self.q, self.k, self.v = (nn.Conv2d(d, d, (1, 11), padding=(0, 5)) for _ in range(3))
        self.fc = nn.Linear(n_bins, 1)

    def forward(self, f):                                           # [B, D, Nb, T]
        q, k, v = self.q(f), self.k(f), self.v(f)
        c = torch.softmax(torch.einsum("bdnt,bdmt->bdnm", q, k) / math.sqrt(q.shape[1] * q.shape[-1]), -1)
        return self.fc(torch.einsum("bdnm,bdmt->bdnt", c, v).permute(0, 1, 3, 2)).squeeze(-1)

def sinusoidal_positions(length: int, d: int) -> torch.Tensor:
    """[length, d] fixed sin / cos positional encoding (Vaswani 2017)."""
    pos = torch.arange(length)[:, None]
    div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe = torch.zeros(length, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe

class MmFormer(nn.Module):
    """`positional` adds the positional encoding the Transformer needs to know where in the window
    a beat is (default False only so models trained before 2026-10 still load unchanged)."""

    def __init__(self, n_bins: int = 1, widths=(16, 32, 64, 128), d_model: int = 128, layers: int = 4,
                 heads: int = 8, n_experts: int = 5, positional: bool = False):
        super().__init__()
        self.positional = positional
        self.iq = nn.Conv2d(2, widths[0], 1)
        self.enc = nn.ModuleList([_block(widths[max(i - 1, 0)], w) for i, w in enumerate(widths)])
        self.down = nn.ModuleList([_block(w, w, 2) for w in widths])
        self.shortcut = nn.ModuleList([SpatialAttentionShortcut(w, n_bins) for w in widths])
        self.bottleneck = _block(widths[-1], d_model)
        self.bins = nn.Linear(n_bins, 1)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, 0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers)
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        c_in = d_model
        for w in widths[::-1]:
            self.up.append(nn.ConvTranspose1d(c_in, w, 2, 2))
            self.dec.append(nn.Sequential(nn.Conv1d(2 * w, w, 11, padding=5), nn.InstanceNorm1d(w, affine=True),
                                          nn.LeakyReLU(0.3)))
            c_in = w
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Conv1d(widths[0], 64, 11, padding=5), nn.LeakyReLU(0.3),
            nn.Conv1d(64, 32, 11, padding=5), nn.LeakyReLU(0.3),
            nn.Conv1d(32, 16, 11, padding=5), nn.LeakyReLU(0.3),
            nn.Conv1d(16, 1, 11, padding=5)) for _ in range(n_experts)])

    def features(self, x):                                          # [B, 2, Nb, T]
        h, skips = self.iq(x), []
        for enc, down, sc in zip(self.enc, self.down, self.shortcut):
            h = enc(h)
            skips.append(sc(h))
            h = down(h)
        h = self.bins(self.bottleneck(h).permute(0, 1, 3, 2)).squeeze(-1)       # [B, D, T/16]
        h = h.transpose(1, 2)
        if self.positional:
            h = h + sinusoidal_positions(h.shape[1], h.shape[2]).to(h.device)
        h = self.transformer(h).transpose(1, 2)
        for up, dec, skip in zip(self.up, self.dec, skips[::-1]):
            h = dec(torch.cat([up(h), skip], 1))
        return h                                                    # [B, 16, T]

    def forward(self, x, category=None):
        """ABP waveform [B, T] in mmHg (offset from the training mean). category [B] selects the
        expert (Type I); None averages all experts (no cuff information)."""
        f = self.features(x)
        outs = torch.stack([e(f).squeeze(1) for e in self.experts], 1)          # [B, E, T]
        if category is None:
            return outs.mean(1)
        return outs[torch.arange(len(x)), category]

def pearson(a, b):
    a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-8)

def loss_fn(pred, target, alpha: float = 1.0, beta: float = 100.0):
    return alpha * nn.functional.mse_loss(pred, target) + beta * (1 - pearson(pred, target)).mean()


# ------------------------------------------------------------------ training interface
from mmwave_bp import data                                          # noqa: E402

TARGET = "waveform"
TRAIN_STEP_S = 2.0
TEST_STEP_S = WINDOW / FS
OPTIM = {"name": "adam", "lr": 2e-4, "step": None}                  # cosine schedule
BATCH = 32
FLIP_OK = True


def examples(rec: data.Recording, step_s: float):
    """X [m, 2, 1, 1024], Y [m, 1024] BP waveform (mmHg), BP [m, 2], start sample at data.FS.
    Needs a continuous BP reference: recordings with only a cuff reading give no examples."""
    empty = (np.zeros((0, 2, 1, WINDOW), np.float32), np.zeros((0, WINDOW), np.float32),
             np.zeros((0, 2), np.float32), np.zeros(0, int))
    if rec.abp is None:
        return empty
    if rec.pulse.ndim != 1:
        raise ValueError("WaveBP here takes one pulse signal per recording (N_b = 1)")
    channels = radar_channels(rec.pulse, data.FS)
    abp = resample(rec.abp, data.FS, FS)
    n = min(channels.shape[1], len(abp))
    xs, ys, bps, where = [], [], [], []
    for s, x, y in windows(channels[:, :n], max(1, int(round(step_s * FS))), abp[:n]):
        if not data.valid_abp(y, FS):
            continue
        sbp, dbp = beat_values(y)
        xs.append(x); ys.append(y); bps.append((np.median(sbp), np.median(dbp)))
        where.append(int(s * data.FS / FS))
    if not xs:
        return empty
    return np.array(xs, np.float32), np.array(ys, np.float32), np.array(bps, np.float32), np.array(where, int)


def build(x_shape, **kwargs) -> nn.Module:
    return MmFormer(n_bins=x_shape[2], **kwargs)


def categories(bp) -> torch.Tensor:
    """AHA category of every (SBP, DBP) row, as the expert index."""
    return torch.tensor([aha_category(float(s), float(d)) for s, d in bp], dtype=torch.long, device=bp.device)


def loss(model, x, y, bp):
    """Each window trains the expert of its own AHA category (Type I)."""
    return loss_fn(model(x, categories(bp)), y)


def predict(model, x, category: int | None = None):
    cat = None if category is None else torch.full((len(x),), category, dtype=torch.long, device=x.device)
    return model(x, cat)


def to_bp(waveforms: np.ndarray) -> np.ndarray:
    """[m, T] predicted BP waveforms in mmHg -> [m, 2] SBP / DBP."""
    return np.array([bp_from_waveform(w) for w in waveforms], np.float32).reshape(-1, 2)
