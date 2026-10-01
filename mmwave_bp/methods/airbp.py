"""airBP (Liang et al., ACM TIoT 4(4):28, 2023): 30 s wrist pulse -> pre-trained encoder -> BP.

Paper: Infineon BGT60TR24 (60 GHz), wrist 12 cm in front of the radar, 210 Hz slow time,
41 subjects, wrist-cuff reference, leave-one-subject-out.
Front end: amplitude (RSS) of the range / angle cell with the highest autocorrelation, negated
(RSS falls when the artery dilates); VMD with K = 9, pulse = modes 2 + 3 (by centre frequency).
Model: (1) masked auto-encoder pre-training, 50 % contiguous mask, input 6300 samples, encoder =
PreBlock + 16 ResBlocks (32 -> 256 channels) + attention, decoder = 5 deconvolutions, MSE;
(2) frozen encoder -> 3 convolutional self-attention layers (8 heads, kernel 5) -> max-pool
(256) -> [+ z-scored gender, age, height, weight] -> FC -> (SBP, DBP, HR). SGD 1e-3, batch 20,
300 epochs. The paper reports 2.6x / 3.1x larger MAE without the pre-training.

What a public data set gives: one pulse signal per recording, so the cell selection is not part
of this file (do it when you export your own radar data: `dsp.periodicity` is the ACC score).
Ours (not in the paper): block layout (4 stages x 4 blocks, stride 2 per stage), kernel 7,
SGD momentum 0.9, the attention of Eq. 16-17 applied as a temporal gate. Demographic inputs
are off by default (`n_meta = 0`); the training loop does not feed them.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from mmwave_bp import data, dsp

TARGET = "values"
FS = 210.0
LENGTH = 6300                 # 30 s
TRAIN_STEP_S = 10.0
TEST_STEP_S = 30.0
OPTIM = {"name": "sgd", "lr": 1e-3, "step": None}
BATCH = 20
FLIP_OK = False               # the sign carries meaning: the pulse is -RSS
PRETRAIN = True
USE_VMD = True                # set False for a 0.7-8 Hz band-pass instead (much faster to prepare)


def pulse_of(x: np.ndarray) -> np.ndarray:
    """VMD K = 9, modes 2 + 3 by centre frequency; z-scored."""
    if USE_VMD:
        modes, _ = dsp.vmd(x - x.mean(), FS, k=9, tol=1e-6, max_iter=200)
        z = modes[1] + modes[2]
    else:
        z = dsp.bandpass(x, FS, (0.7, 8.0))
    return (z - z.mean()) / (z.std() + 1e-12)


def examples(rec: data.Recording, step_s: float):
    """X [m, 1, 6300], Y [m, 3] (SBP, DBP, pulse rate bpm), BP [m, 2], start sample at data.FS."""
    if rec.pulse.ndim != 1:
        raise ValueError("airBP takes one pulse signal per recording")
    x = dsp.resample(rec.pulse, data.FS, FS)
    xs, ys, where = [], [], []
    for s in range(0, len(x) - LENGTH + 1, max(1, int(step_s * FS))):
        a, b = int(s * data.FS / FS), int((s + LENGTH) * data.FS / FS)
        bp = data.label(rec, a, b)
        if bp is None:
            continue
        z = pulse_of(x[s: s + LENGTH])
        xs.append(z[None])
        ys.append((*bp, 60 * dsp.dominant_rate_hz(z, FS)))
        where.append(a)
    x = np.array(xs, np.float32).reshape(-1, 1, LENGTH)
    y = np.array(ys, np.float32).reshape(-1, 3)
    return x, y, y[:, :2], np.array(where, int)


class _ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, k: int = 7):
        super().__init__()
        self.main = nn.Sequential(nn.Conv1d(cin, cout, k, stride, k // 2), nn.BatchNorm1d(cout), nn.ReLU(),
                                  nn.Conv1d(cout, cout, k, 1, k // 2), nn.BatchNorm1d(cout))
        self.shortcut = (nn.Identity() if cin == cout and stride == 1
                         else nn.Sequential(nn.Conv1d(cin, cout, 1, stride), nn.BatchNorm1d(cout)))

    def forward(self, x):
        return torch.relu(self.main(x) + self.shortcut(x))


class Encoder(nn.Module):
    """PreBlock + 16 ResBlocks (32, 64, 128, 256) + attention; time is reduced 32 times."""

    def __init__(self, widths=(32, 64, 128, 256), blocks: int = 4):
        super().__init__()
        self.pre = nn.Sequential(nn.Conv1d(1, widths[0], 15, 2, 7), nn.BatchNorm1d(widths[0]), nn.ReLU())
        layers, cin = [], widths[0]
        for w in widths:
            layers += [_ResBlock(cin if i == 0 else w, w, 2 if i == 0 else 1) for i in range(blocks)]
            cin = w
        self.blocks = nn.Sequential(*layers)
        self.score = nn.Conv1d(widths[-1], 1, 1)                     # W of Eq. 16

    def forward(self, x):                                            # [B, 1, L] -> [B, 256, L / 32]
        y = self.blocks(self.pre(x))
        gate = torch.softmax(self.score(torch.tanh(y)), dim=-1)      # Eq. 16-17
        return y * gate * y.shape[-1]


class _ConvSelfAttention(nn.Module):
    """Eq. 19 with Q, K, V from convolutions (kernel 5), 8 heads, residual + LayerNorm."""

    def __init__(self, d: int = 256, heads: int = 8, k: int = 5):
        super().__init__()
        self.heads = heads
        self.q, self.k, self.v = (nn.Conv1d(d, d, k, padding=k // 2) for _ in range(3))
        self.out, self.norm = nn.Conv1d(d, d, 1), nn.LayerNorm(d)

    def forward(self, x):                                            # [B, D, T]
        b, d, t = x.shape
        q, k, v = (m(x).view(b, self.heads, d // self.heads, t) for m in (self.q, self.k, self.v))
        w = torch.softmax(torch.einsum("bhdi,bhdj->bhij", q, k) / math.sqrt(d // self.heads), -1)
        y = self.out(torch.einsum("bhij,bhdj->bhdi", w, v).reshape(b, d, t))
        return self.norm((x + y).transpose(1, 2)).transpose(1, 2)


class AirBP(nn.Module):
    def __init__(self, d: int = 256, n_meta: int = 0, outputs: int = 3):
        super().__init__()
        self.encoder = Encoder()
        chans = (d, 128, 64, 32, 16)
        deconv = []
        for i in range(4):
            deconv += [nn.ConvTranspose1d(chans[i], chans[i + 1], 4, 2, 1), nn.ReLU()]
        self.decoder = nn.Sequential(*deconv, nn.ConvTranspose1d(chans[-1], 1, 4, 2, 1))     # 5 deconvolutions
        self.estimator = nn.Sequential(*[_ConvSelfAttention(d) for _ in range(3)])
        self.head = nn.Linear(d + n_meta, outputs)

    @staticmethod
    def _pad(x):
        return nn.functional.pad(x, (0, -x.shape[-1] % 32))

    def reconstruct(self, x):
        return self.decoder(self.encoder(self._pad(x)))[..., : x.shape[-1]]

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def forward(self, x, meta=None):
        frozen = not next(self.encoder.parameters()).requires_grad
        if frozen:
            self.encoder.eval()                                      # frozen: BatchNorm statistics too
        h = self.estimator(self.encoder(self._pad(x))).amax(-1)
        return self.head(h if meta is None else torch.cat([h, meta], 1))


def build(x_shape, **kwargs) -> nn.Module:
    return AirBP(**kwargs)


def pretrain_loss(model, x, masked: float = 0.5):
    """Hide one contiguous half of every signal, reconstruct the whole signal (MSE)."""
    n = x.shape[-1]
    width = int(masked * n)
    start = torch.randint(0, n - width + 1, (len(x),), device=x.device)
    t = torch.arange(n, device=x.device)[None]
    keep = ((t < start[:, None]) | (t >= start[:, None] + width)).unsqueeze(1)
    return nn.functional.mse_loss(model.reconstruct(x * keep), x)


def after_pretrain(model) -> None:
    model.freeze_encoder()


def loss(model, x, y, bp):
    return nn.functional.mse_loss(model(x), y)


def predict(model, x):
    return model(x)
