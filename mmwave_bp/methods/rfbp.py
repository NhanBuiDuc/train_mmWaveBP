"""RF-BP (Wang et al., IMWUT 8(2):65, 2024): quasi-pulse windows -> multi-scale attention ResNet.

Paper: Novelda X4M05 UWB at 200 frames/s, chest at 30-60 cm, 70 subjects, cuff reference.
Pipeline: 20 s segments kept when stationary (autocorrelation > 0.6), 2nd-order zero-phase
Butterworth 0.8-10 Hz "pseudo pulse", AMPD peaks, aliased-peak rejection (spacing vs the
spectral pulse period), deformed-peak rejection (cosine similarity with the first principal
component < 0.8), +-4 s windows around each kept peak, channels [pulse, 1st derivative].
Network: 4 parallel convs (k = 5, 9, 13, 17) -> 8 pre-activation residual blocks with channel
attention (64, 64, 128, 128, 256, 256, 512, 512) -> global average pool -> FC -> (SBP, DBP).
Loss: 1.5 * Huber(SBP) + Huber(DBP), delta = 15. Adam 1e-3, x0.8 every 5 epochs, batch 32.

Ours (not given in the paper): block kernel 3, stride 2 where the width changes, attention
reduction 16, per-window z-score, the autocorrelation lag range, the aliasing tolerance (25 %).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from mmwave_bp import data, dsp

TARGET = "values"
FS = 200.0
HALF_WINDOW_S = 4.0
SEGMENT_S = 20.0
TRAIN_STEP_S = 2.0            # minimum spacing of the peaks used for training
TEST_STEP_S = 8.0
OPTIM = {"name": "adam", "lr": 1e-3, "step": (5, 0.8)}
BATCH = 32
FLIP_OK = True


def _zscore(w: np.ndarray) -> np.ndarray:
    return (w - w.mean()) / (w.std() + 1e-12)


def examples(rec: data.Recording, step_s: float):
    """X [m, 2, 1600], Y [m, 2] (SBP, DBP), BP [m, 2], start sample at data.FS [m]."""
    if rec.pulse.ndim != 1:
        raise ValueError("RF-BP takes one pulse signal per recording")
    x = dsp.resample(rec.pulse, data.FS, FS)
    pulse = dsp.bandpass(x, FS, (0.8, 10.0), order=2)
    seg, half = int(SEGMENT_S * FS), int(HALF_WINDOW_S * FS)
    xs, bps, where, last = [], [], [], -np.inf
    for start in range(0, len(pulse) - seg + 1, seg):
        r = pulse[start: start + seg]
        if dsp.stationarity(x[start: start + seg], FS) <= 0.6:
            continue
        peaks = dsp.ampd(r, max_scale=int(1.5 * FS))
        if len(peaks) < 3:
            continue
        period = FS / dsp.dominant_rate_hz(r, FS, (0.8, 3.0))
        ok = dsp.interval_ok(peaks, period) & (np.nan_to_num(dsp.pca_similarity(r, peaks, int(period / 2))) >= 0.8)
        for q in peaks[ok] + start:
            if q - half < 0 or q + half > len(pulse) or q - last < step_s * FS:
                continue
            a, b = int((q - half) * data.FS / FS), int((q + half) * data.FS / FS)
            bp = data.label(rec, a, b)
            if bp is None:
                continue
            w = pulse[q - half: q + half]
            xs.append(np.stack([_zscore(w), _zscore(np.gradient(w))]))
            bps.append(bp)
            where.append(a)
            last = q
    x = np.array(xs, np.float32).reshape(-1, 2, 2 * half)
    bp = np.array(bps, np.float32).reshape(-1, 2)
    return x, bp, bp, np.array(where, int)


class _ChannelAttention(nn.Module):
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        h = max(c // r, 4)
        self.mlp = nn.Sequential(nn.Conv1d(c, h, 1), nn.ReLU(), nn.Conv1d(h, c, 1))

    def forward(self, x):
        return torch.sigmoid(self.mlp(x.mean(-1, keepdim=True)) + self.mlp(x.amax(-1, keepdim=True)))


class _AttentionResBlock(nn.Module):
    """Pre-activation residual block: (BN, ReLU, conv) x 2, channel attention, no final ReLU."""

    def __init__(self, cin: int, cout: int, stride: int, k: int = 3):
        super().__init__()
        self.main = nn.Sequential(nn.BatchNorm1d(cin), nn.ReLU(), nn.Conv1d(cin, cout, k, stride, k // 2),
                                  nn.BatchNorm1d(cout), nn.ReLU(), nn.Conv1d(cout, cout, k, 1, k // 2))
        self.attention = _ChannelAttention(cout)
        self.shortcut = nn.Sequential(nn.Conv1d(cin, cout, 1, stride), nn.BatchNorm1d(cout))

    def forward(self, x):
        y = self.main(x)
        return self.shortcut(x) + y * self.attention(y)


class RFBPNet(nn.Module):
    def __init__(self, widths: tuple[int, ...] = (64, 64, 128, 128, 256, 256, 512, 512)):
        super().__init__()
        self.scales = nn.ModuleList([nn.Sequential(nn.Conv1d(2, 8, k, padding=k // 2), nn.BatchNorm1d(8), nn.ReLU())
                                     for k in (5, 9, 13, 17)])
        chans = (32, *widths)
        self.encoder = nn.Sequential(*[_AttentionResBlock(chans[i], chans[i + 1], 2 if chans[i + 1] != chans[i] else 1)
                                       for i in range(len(widths))])
        self.head = nn.Linear(widths[-1], 2)

    def forward(self, x):                                   # [B, 2, L]
        return self.head(self.encoder(torch.cat([m(x) for m in self.scales], 1)).mean(-1))


def build(x_shape, **kwargs) -> nn.Module:
    return RFBPNet(**kwargs)


def loss(model, x, y, bp, delta: float = 15.0, sbp_weight: float = 1.5):
    out = model(x)
    huber = nn.functional.huber_loss
    return sbp_weight * huber(out[:, 0], y[:, 0], delta=delta) + huber(out[:, 1], y[:, 1], delta=delta)


def predict(model, x):
    return model(x)
