"""Training and subject-independent evaluation, shared by every method.

A method module (mmwave_bp/methods/*.py) provides
    examples(recording, step_s) -> X, Y, BP [m, 2], start [m]
    build(x_shape, **kwargs) -> torch module          (or fit / predict for KIND = "sklearn")
    loss(model, x, y, bp) -> scalar,  predict(model, x) -> output in the space of Y
    TARGET "values" (Y[:, :2] = SBP, DBP) or "waveform" (to_bp turns outputs into SBP, DBP)
    optional: pretrain_loss + after_pretrain, OPTIM, BATCH, FLIP_OK, TRAIN_STEP_S, TEST_STEP_S

Evaluation is always on people the model never saw (folds of whole subjects). Per test subject
the first `cal` windows (resting first, then in time) are calibration and are excluded from
every score. Rows of the report:
    model                 the network as it is
    model + offset        its mean error on the calibration windows removed (one cuff reading)
    model, Type I         WaveBP only: the expert of the calibration's AHA category
    B1 population mean    the training subjects' mean BP for everybody
    B0 carry calibration  the calibration value repeated
A method is useful only when it beats B1 (without calibration) or B0 (with calibration).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import pickle
import time

import numpy as np

from mmwave_bp import metrics as ev


@dataclass
class TrainConfig:
    epochs: int = 50
    pretrain_epochs: int = 0
    lr: float | None = None       # None: the method's own
    batch: int | None = None
    seed: int = 0
    flip: bool = False            # random sign flip per window (the sign of radar displacement depends on the radar)
    val_fraction: float = 0.15    # training subjects set aside to pick the best epoch (0 = keep the last)
    patience: int = 0             # stop after this many epochs without a better validation loss (0 = never)
    device: str = "auto"


@dataclass
class Examples:
    x: np.ndarray
    y: np.ndarray
    bp: np.ndarray                # [m, 2] reference SBP, DBP
    subject: np.ndarray
    rank: np.ndarray              # sort key within a subject: resting first, then in time

    def take(self, mask) -> "Examples":
        return Examples(self.x[mask], self.y[mask], self.bp[mask], self.subject[mask], self.rank[mask])

    def __len__(self) -> int:
        return len(self.x)


def make_examples(method, recordings, step_s: float, cache: Path | None = None, log=print) -> Examples:
    """Cut every recording into the method's windows (cached as one .npz when `cache` is given)."""
    if cache is not None and Path(cache).exists():
        d = np.load(cache, allow_pickle=False)
        return Examples(d["x"], d["y"], d["bp"], d["subject"], d["rank"])
    xs, ys, bps, subjects, ranks = [], [], [], [], []
    started = time.perf_counter()
    for i, rec in enumerate(recordings):
        x, y, bp, start = method.examples(rec, step_s)
        if len(x):
            xs.append(x); ys.append(y); bps.append(bp)
            subjects += [rec.subject] * len(x)
            ranks.append(rec.order * 1e9 + i * 1e7 + start)
        if log and (i + 1) % 20 == 0:
            log(f"  windows: {i + 1}/{len(recordings)} recordings ({time.perf_counter() - started:.0f} s)")
    if not xs:
        raise SystemExit("no usable window: this method cannot use this data set (see the method's docstring)")
    out = Examples(np.concatenate(xs), np.concatenate(ys), np.concatenate(bps), np.array(subjects),
                   np.concatenate(ranks))
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, x=out.x, y=out.y, bp=out.bp, subject=out.subject, rank=out.rank)
    return out


# ------------------------------------------------------------------ torch training
def _device(name: str):
    import torch
    return torch.device(("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name)


def _save_atomic(obj, path: Path) -> None:
    """Write to a temporary file, then rename: an interrupted save never corrupts a checkpoint."""
    import torch
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _target_mean(method, y: np.ndarray) -> np.ndarray:
    return np.float32(y.mean()) if method.TARGET == "waveform" else y.mean(axis=0).astype(np.float32)


def forward(method, net, x: np.ndarray, batch: int = 64, **kwargs) -> np.ndarray:
    """Raw outputs (offset from the training mean) of a net in eval mode."""
    import torch
    device = next(net.parameters()).device
    was_training = net.training
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(method.predict(net, torch.as_tensor(x[i: i + batch]).to(device), **kwargs).cpu().numpy())
    net.train(was_training)
    return np.concatenate(out)


def _epoch(method, net, opt, sched, xt, yt, bt, loss_fn, config, device, per_step: bool) -> float:
    import torch
    net.train()
    batch = config.batch or getattr(method, "BATCH", 32)
    order, total = torch.randperm(len(xt)), 0.0
    for i in range(0, len(xt), batch):
        idx = order[i: i + batch]
        if len(idx) < 2:                                              # BatchNorm cannot take one sample
            continue
        xb = xt[idx].to(device)
        if config.flip and getattr(method, "FLIP_OK", False):
            xb = xb * (torch.randint(0, 2, (len(idx),) + (1,) * (xb.dim() - 1), device=device) * 2 - 1)
        opt.zero_grad()
        loss = loss_fn(net, xb, yt[idx].to(device), bt[idx].to(device))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"loss became {float(loss)}: check the inputs for NaN / Inf "
                                     "(the last checkpoint is still valid)")
        loss.backward()
        opt.step()
        if per_step:
            sched.step()
        total += float(loss.detach()) * len(idx)
    if not per_step:
        sched.step()
    return total / len(xt)


def _validation_loss(method, net, val: Examples, mean, device) -> float:
    """Mean absolute error of SBP and DBP (mmHg) on the validation subjects, no calibration."""
    pred = to_bp(method, forward(method, net, val.x) + mean)
    return float(np.nanmean(np.abs(pred - val.bp)))


def _optimiser(method, params, config, steps_per_epoch: int, epochs: int):
    import torch
    spec = getattr(method, "OPTIM", {"name": "adam", "lr": 1e-3, "step": None})
    lr = config.lr or spec["lr"]
    opt = (torch.optim.SGD(params, lr=lr, momentum=0.9) if spec["name"] == "sgd"
           else torch.optim.Adam(params, lr=lr))
    if spec.get("step"):
        return opt, torch.optim.lr_scheduler.StepLR(opt, spec["step"][0], spec["step"][1]), False
    total = max(1, epochs * steps_per_epoch)
    return opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, total, eta_min=lr / 20), True


def train(method, name: str, tr: Examples, config: TrainConfig, model_kwargs: dict | None = None,
          checkpoint: Path | None = None, log=print) -> dict:
    """Returns a bundle {method, kwargs, x_shape, state, target_mean, epoch, history}.

    With `checkpoint`, the full state (weights, optimiser, schedule, RNG, epoch, best weights) is
    written after every epoch and training resumes from it when the file exists; resuming with
    other data or settings is refused."""
    model_kwargs = model_kwargs or {}
    if getattr(method, "KIND", "torch") == "sklearn":
        model = method.fit(tr.x, tr.y, **model_kwargs)
        return {"method": name, "kind": "sklearn", "kwargs": model_kwargs, "model": model}
    import torch
    torch.manual_seed(config.seed)
    device = _device(config.device)
    val = None
    people = sorted(set(tr.subject))
    if config.val_fraction > 0 and len(people) >= 4:
        k = max(1, int(round(config.val_fraction * len(people))))
        chosen = np.random.default_rng(config.seed).permutation(people)[:k]
        mask = np.isin(tr.subject, chosen)
        val, tr = tr.take(mask), tr.take(~mask)
    mean = _target_mean(method, tr.y)
    net = method.build(tr.x.shape, **model_kwargs).to(device)
    xt, yt, bt = torch.as_tensor(tr.x), torch.as_tensor(tr.y - mean), torch.as_tensor(tr.bp)
    steps = math.ceil(len(xt) / (config.batch or getattr(method, "BATCH", 32)))
    pre = config.pretrain_epochs if hasattr(method, "pretrain_loss") else 0
    signature = {"n": len(xt), "config": {k: v for k, v in asdict(config).items() if k != "device"},
                 "model": model_kwargs}
    state = {"epoch": 0, "best": None, "best_loss": float("inf"), "best_epoch": 0, "history": []}
    stages = {}

    def stage(tag: str, epochs: int):
        if tag not in stages:
            params = [p for p in net.parameters() if p.requires_grad]
            stages[tag] = _optimiser(method, params, config, steps, epochs)
        return stages[tag]

    if checkpoint is not None and Path(checkpoint).exists():
        ck = torch.load(checkpoint, weights_only=False, map_location="cpu")
        if ck["signature"] != signature:
            raise SystemExit(f"{checkpoint} was made with other data or settings: delete it or use --fresh")
        net.load_state_dict(ck["model"])
        state = ck["state"]
        if state["epoch"] >= pre and pre:
            method.after_pretrain(net)
        tag = "pretrain" if state["epoch"] < pre else "main"
        opt, sched, _ = stage(tag, pre if tag == "pretrain" else config.epochs)
        opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        torch.set_rng_state(ck["rng"])
        if log:
            log(f"    resuming from {Path(checkpoint).name}: {state['epoch']}/{pre + config.epochs} epochs done")
    started = time.perf_counter()
    stale = 0
    for epoch in range(state["epoch"], pre + config.epochs):
        pretraining = epoch < pre
        if epoch == pre and pre:
            method.after_pretrain(net)
        opt, sched, per_step = stage("pretrain" if pretraining else "main", pre if pretraining else config.epochs)
        fn = (lambda m, x, y, b: method.pretrain_loss(m, x)) if pretraining else method.loss
        train_loss = _epoch(method, net, opt, sched, xt, yt, bt, fn, config, device, per_step)
        note = ""
        if not pretraining:
            if val is not None and len(val):
                v = _validation_loss(method, net, val, mean, device)
                note = f" | validation MAE {v:.2f} mmHg"
                if v < state["best_loss"]:
                    state.update(best={k: t.detach().cpu().clone() for k, t in net.state_dict().items()},
                                 best_loss=v, best_epoch=epoch + 1 - pre)
                    stale, note = 0, note + " (best)"
                else:
                    stale += 1
                state["history"].append({"epoch": epoch + 1 - pre, "train_loss": train_loss, "validation_mae": v})
            else:
                state["history"].append({"epoch": epoch + 1 - pre, "train_loss": train_loss})
        state["epoch"] = epoch + 1
        if checkpoint is not None:
            _save_atomic({"signature": signature, "state": state, "model": net.state_dict(),
                          "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                          "rng": torch.get_rng_state()}, Path(checkpoint))
        if log:
            tag = f"pretrain {epoch + 1}/{pre}" if pretraining else f"epoch {epoch + 1 - pre}/{config.epochs}"
            log(f"    {tag}: loss {train_loss:.3f} ({time.perf_counter() - started:.0f} s){note}")
        if config.patience and stale >= config.patience:
            if log:
                log(f"    stopped: no better validation loss for {stale} epochs")
            break
    weights = state["best"] if state["best"] is not None else {k: t.detach().cpu() for k, t in net.state_dict().items()}
    return {"method": name, "kind": "torch", "kwargs": model_kwargs, "x_shape": tuple(tr.x.shape),
            "state": weights, "target_mean": mean, "epoch": state["best_epoch"] or state["epoch"] - pre,
            "history": state["history"]}


def load(method, bundle: dict, device: str = "auto"):
    """The trained torch module of a bundle, in eval mode."""
    net = method.build(bundle["x_shape"], **bundle["kwargs"])
    net.load_state_dict(bundle["state"])
    return net.to(_device(device)).eval()


def to_bp(method, outputs: np.ndarray) -> np.ndarray:
    return method.to_bp(outputs) if method.TARGET == "waveform" else np.asarray(outputs)[:, :2]


def predictor(method, bundle: dict, device: str = "auto"):
    """f(x, **kwargs) -> [m, 2] SBP / DBP in mmHg."""
    if bundle["kind"] == "sklearn":
        return lambda x, **kw: np.asarray(method.predict(bundle["model"], x))[:, :2]
    net = load(method, bundle, device)
    return lambda x, **kw: to_bp(method, forward(method, net, x, **kw) + bundle["target_mean"])


# ------------------------------------------------------------------ evaluation
def evaluate(method, bundle: dict, te: Examples, cal: int, population_mean: np.ndarray, device: str = "auto"):
    """({row: [n, 2]}, reference [n, 2], subject [n]) on unseen people; see the module docstring."""
    predict = predictor(method, bundle, device)
    type1 = hasattr(method, "aha_category") and bundle["kind"] == "torch"
    rows = {"model": [], "model + offset": [], "B1 population mean": [], "B0 carry calibration": []}
    if type1:
        rows["model, Type I (AHA category)"] = []
    ref, who = [], []
    for s in sorted(set(te.subject)):
        idx = np.flatnonzero(te.subject == s)
        if len(idx) <= cal:
            continue
        idx = idx[np.argsort(te.rank[idx], kind="stable")]
        first, rest = idx[:cal], idx[cal:]
        raw = predict(te.x[idx])
        anchor = te.bp[first].mean(axis=0)
        rows["model"].append(raw[cal:])
        rows["model + offset"].append(raw[cal:] + np.nanmean(te.bp[first] - raw[:cal], axis=0))
        rows["B1 population mean"].append(np.tile(population_mean, (len(rest), 1)))
        rows["B0 carry calibration"].append(np.tile(anchor, (len(rest), 1)))
        if type1:
            rows["model, Type I (AHA category)"].append(predict(te.x[rest], category=method.aha_category(*anchor)))
        ref.append(te.bp[rest])
        who += [s] * len(rest)
    if not ref:
        return {k: np.zeros((0, 2)) for k in rows}, np.zeros((0, 2)), np.zeros(0, str)
    return {k: np.concatenate(v) for k, v in rows.items()}, np.concatenate(ref), np.array(who)


def report(rows: dict, ref: np.ndarray, subjects: np.ndarray, log=print) -> dict:
    """{row: {sbp: metrics, dbp: metrics}} and a printed table."""
    out = {}
    if log:
        log(f"\n{'':<32} {'SBP MAE':>8} {'ME+-SD':>13} {'r':>6}   {'DBP MAE':>8} {'ME+-SD':>13} {'r':>6}")
    for name, pred in rows.items():
        entry = {t: ev.metrics(pred[:, j], ref[:, j], subjects, t) for j, t in enumerate(("sbp", "dbp"))}
        out[name] = entry
        s, d = entry["sbp"], entry["dbp"]
        if log and "mae" in s and "mae" in d:
            log(f"{name:<32} {s['mae']:8.1f} {s['me']:+6.1f}+-{s['sd']:<5.1f} {s['tracking_r']:+6.2f}   "
                f"{d['mae']:8.1f} {d['me']:+6.1f}+-{d['sd']:<5.1f} {d['tracking_r']:+6.2f}")
    if log:
        log("r = within-subject correlation of estimate and reference (does it follow BP changes?)")
    return out


def save_bundle(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if bundle["kind"] == "sklearn":
        with open(path, "wb") as f:
            pickle.dump(bundle, f)
    else:
        _save_atomic(bundle, path)
