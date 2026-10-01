"""hBP-Fi (Cao et al., IEEE INFOCOM 2024): pulse waves at several arm sites -> BP.

Paper: TI xWR1843BOOST + DCA1000 (77 GHz) at 200 frames/s, radar ~50 cm from an arm stretched on
a table, 35 subjects, arm-cuff reference. Range FFT -> differential beamforming with null
steering separates arm sites -> phase per site -> windows of 512 samples, hop 64 -> BP-specific
transfer function (BTF) between sites, Gamma = P_o P_i^-1 -> ADMM-unrolled, weight-shared DRCN
constrained by a tube-load model -> LSTM -> Dropout -> LSTM -> Dropout -> Dense -> (SBP, DBP).
Loss (Eq. 13): ||J(Theta)||^2 + ||A Theta + B Z - C||^2 + ||rho - rho_hat||^2.

THIS FILE IS AN INTERPRETATION, NOT A REPRODUCTION. The paper specifies the network only as a
block diagram: no layer sizes, no number of unrolled iterations, no A / B / C / Z, no step sizes,
no physiological constants, no BTF computation, no input shape, no optimiser, and no rule that
assigns a cuff reading to a window. Its 5-fold split is not stated to be by subject. No code or
data are public. What is taken from the paper: 200 Hz, 512 / 64 windows, BTF inputs, 64
frequency groups (its SHAP analysis), the tube-load transfer function (Eq. 2-4), the cost
J (Eq. 5), the unrolled Theta / Z / lambda updates (Eq. 12), the block order, the loss terms.
Ours: everything else, namely
* BTF = regularised spectral ratio P_i conj(P_0) / (|P_0|^2 + eps), real and imaginary part;
* the standard tube-load delay exp(+-j w tau) with tau = l * sqrt(eta C0) * exp(-alpha rho / 2)
  (the paper's Eq. 4 drops the Laplace variable, which leaves no delay at all);
* A = I, B = -I, C = 0 (Z is a network-refined copy of Theta), 4 iterations, learnt step sizes
  and constants a, b, alpha, sqrt(eta C0);
* 0.5-20 Hz band-pass before the transform, all layer sizes, Adam 1e-3.

Data: `Recording.pulse` must be [k, n] with k >= 2 arm sites (site 0 = proximal). No public
data set provides this with BP labels; the beamforming front end that produces the sites from
raw ADC data belongs to the capture code, not to this training repo.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from mmwave_bp import data, dsp

TARGET = "values"
FS = 200.0
WINDOW = 512
TRAIN_STEP_S = 64 / FS
TEST_STEP_S = WINDOW / FS
OPTIM = {"name": "adam", "lr": 1e-3, "step": None}
BATCH = 64
FLIP_OK = False
BAND_HZ = (0.5, 20.0)


def examples(rec: data.Recording, step_s: float):
    """X [m, k, 512] band-passed, z-scored site signals, Y [m, 2], BP [m, 2], start at data.FS."""
    if rec.pulse.ndim != 2 or rec.pulse.shape[0] < 2:
        raise ValueError("hBP-Fi needs pulse signals of at least 2 arm sites per recording ([k, n])")
    x = dsp.bandpass(dsp.resample(rec.pulse, data.FS, FS), FS, BAND_HZ)
    xs, bps, where = [], [], []
    for s in range(0, x.shape[1] - WINDOW + 1, max(1, int(round(step_s * FS)))):
        a, b = int(s * data.FS / FS), int((s + WINDOW) * data.FS / FS)
        bp = data.label(rec, a, b) if rec.abp is None else data.label(rec, a, max(b, a + int(4 * data.FS)))
        if bp is None:
            continue
        w = x[:, s: s + WINDOW]
        xs.append((w - w.mean(1, keepdims=True)) / (w.std(1, keepdims=True) + 1e-12))
        bps.append(bp); where.append(a)
    bp = np.array(bps, np.float32).reshape(-1, 2)
    return np.array(xs, np.float32).reshape(-1, x.shape[0], WINDOW), bp, bp, np.array(where, int)


class TubeLoad(nn.Module):
    """Inverse transfer function site 0 -> site i and the consistency cost J (Eq. 2-5).

    Psi(s) = a / (s + b) (Eq. 2, simplified), Gamma^-1 = (e^{s tau} + e^{-s tau} Psi) / (1 + Psi).
    Theta = [rho, l_1 .. l_{k-1}]: rho is the pressure variable (mean-subtracted, in 100 mmHg),
    l_i the path lengths. J = mean squared spread of the site-0 waveforms reconstructed from
    every site."""

    def __init__(self, fs: float = FS, n: int = WINDOW, band=BAND_HZ):
        super().__init__()
        f = torch.fft.rfftfreq(n, 1 / fs)
        self.register_buffer("keep", (f >= band[0]) & (f <= band[1]))
        self.register_buffer("omega", 2 * math.pi * f[(f >= band[0]) & (f <= band[1])])
        self.raw = nn.Parameter(torch.tensor([1.0, 3.0, 0.5, -3.0]))   # a, b, alpha, log(sqrt(eta C0)) [s per unit l]

    def cost(self, spectra, theta):                                   # spectra [B, k, F] complex, theta [B, k]
        a, b, alpha = nn.functional.softplus(self.raw[:3])
        s = 1j * self.omega                                           # [F]
        psi = a / (s + b)
        tau = nn.functional.softplus(theta[:, 1:]) * torch.exp(self.raw[3] - alpha * theta[:, :1] / 2)   # [B, k-1]
        phase = s[None, None] * tau[..., None]                        # [B, k-1, F]
        inverse = (torch.exp(phase) + torch.exp(-phase) * psi) / (1 + psi)
        p = spectra / (spectra.abs().pow(2).sum(-1, keepdim=True).sqrt() + 1e-9)
        estimates = torch.cat([p[:, :1], inverse * p[:, 1:]], 1)      # site-0 waveform seen from every site
        return (estimates - estimates.mean(1, keepdim=True)).abs().pow(2).sum(-1).mean(1)          # [B]


class HBPFi(nn.Module):
    def __init__(self, sites: int = 3, iterations: int = 4, channels: int = 32, recursions: int = 3,
                 hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.iterations, self.recursions = iterations, recursions
        self.physics = TubeLoad()
        self.embed = nn.Conv1d(2 * (sites - 1), channels, 5, padding=2)
        self.recursive = nn.Sequential(nn.Conv1d(channels, channels, 5, padding=2), nn.ReLU())   # shared weights
        self.pool = nn.AdaptiveAvgPool1d(64)                          # 64 frequency groups
        self.init_theta = nn.Linear(channels, sites)
        self.refine = nn.Sequential(nn.Linear(channels + sites, hidden), nn.ReLU(), nn.Linear(hidden, sites))
        self.step = nn.Parameter(torch.tensor(0.1))
        self.penalty = nn.Parameter(torch.tensor(0.5))
        self.lstm1 = nn.LSTM(channels + sites, hidden, batch_first=True)
        self.lstm2 = nn.LSTM(hidden, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden, 2)
        self.aux = None                                               # (J, constraint residual) of the last forward

    def btf(self, spectra):
        """Measured BTFs site 0 -> site i as [B, 2 (k - 1), F]: real and imaginary part of the
        regularised ratio, compressed to |.| < 1."""
        p0 = spectra[:, :1]
        power = p0.abs().pow(2)
        h = spectra[:, 1:] * p0.conj() / (power + 0.01 * power.mean(-1, keepdim=True) + 1e-9)
        h = h / (1 + h.abs())
        return torch.cat([h.real, h.imag], 1)

    def forward(self, x):                                             # [B, k, 512]
        spectra = torch.fft.rfft(x, dim=-1)[..., 1: x.shape[-1] // 2 + 1]
        h = torch.relu(self.embed(self.btf(spectra)))
        for _ in range(self.recursions):                              # deeply-recursive: same layer each time
            h = h + self.recursive(h)
        h = self.pool(h)                                              # [B, C, 64]
        summary = h.mean(-1)
        band = spectra[..., self.physics.keep[1: x.shape[-1] // 2 + 1]]
        with torch.enable_grad():                                     # the unrolled step needs dJ/dTheta
            theta = self.init_theta(summary)
            if not theta.requires_grad:
                theta = theta.requires_grad_(True)
            z, lam = theta, torch.zeros_like(theta)
            for _ in range(self.iterations):                          # Eq. 12
                j = self.physics.cost(band, theta)
                grad, = torch.autograd.grad(j.sum(), theta, create_graph=self.training)
                theta = theta - self.step * grad - self.penalty * (theta - z + lam)
                z = self.refine(torch.cat([summary, theta + lam], 1))
                lam = lam + theta - z
            j = self.physics.cost(band, theta)
        self.aux = (j, (theta - z).pow(2).sum(1))
        seq = torch.cat([h.transpose(1, 2), z[:, None].expand(-1, h.shape[-1], -1)], -1)
        out, _ = self.lstm1(seq)
        out, _ = self.lstm2(self.drop(out))
        return self.dense(self.drop(out[:, -1]))


def build(x_shape, **kwargs) -> nn.Module:
    return HBPFi(sites=x_shape[1], **kwargs)


def loss(model, x, y, bp, physics_weight: float = 1.0, constraint_weight: float = 1.0):
    out = model(x)
    j, residual = model.aux
    return (nn.functional.mse_loss(out, y) + physics_weight * j.pow(2).mean()
            + constraint_weight * residual.mean())


def predict(model, x):
    return model(x).detach()
