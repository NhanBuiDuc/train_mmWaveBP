"""Check the installation: every method trains for one epoch on synthetic pulses and is scored.

    python selftest.py            (about a minute on a CPU; mmbp is skipped without scikit-learn)

The data are synthetic (a pulse train whose shape depends on a made-up BP), so the numbers mean
nothing: the test only shows that windows, model, loss, checkpoint and evaluation run.
"""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mmwave_bp import data, engine, methods                         # noqa: E402


def synthetic(subject: int, seconds: float, sites: int, continuous: bool, rng) -> data.Recording:
    n = int(seconds * data.FS)
    t = np.arange(n) / data.FS
    sbp, dbp = 100 + 8 * subject + rng.normal(0, 2), 65 + 4 * subject + rng.normal(0, 2)
    rate = 1.0 + 0.05 * subject
    phase = 2 * np.pi * rate * t
    beat = lambda p, k: np.exp(k * (np.cos(p) - 1))                  # noqa: E731  one pulse per period
    pulse = np.array([beat(phase - 0.3 * i * (1.2 - sbp / 200), 6) + 0.3 * beat(phase - 2.0 - 0.3 * i, 3)
                      + 0.5 * np.sin(2 * np.pi * 0.25 * t) + 0.02 * rng.normal(size=n) for i in range(sites)])
    rec = data.Recording(f"S{subject:02d}", "rest", "chest", (pulse if sites > 1 else pulse[0]).astype(np.float32))
    if continuous:
        rec.abp = (dbp + (sbp - dbp) * beat(phase - 0.4, 4)).astype(np.float32)
    else:
        rec.sbp, rec.dbp = float(sbp), float(dbp)
    return rec


def main() -> int:
    rng = np.random.default_rng(0)
    failed = []
    for name in methods.NAMES:
        method = methods.get(name)
        sites = 3 if name == "hbpfi" else 1
        recs = [synthetic(s, 100.0, sites, continuous=name in ("wavebp", "rfbp"), rng=rng) for s in range(6)]
        if name == "airbp":
            method.USE_VMD = False                                   # keep the test short
        try:
            tr = engine.make_examples(method, recs, method.TRAIN_STEP_S * 4, log=None)
            te = engine.make_examples(method, recs, method.TEST_STEP_S, log=None)
            test = ["S04", "S05"]
            config = engine.TrainConfig(epochs=1, pretrain_epochs=1, val_fraction=0.25, device="cpu")
            kwargs = {"widths": (8, 8, 16, 16), "d_model": 16, "layers": 1, "heads": 2} if name == "wavebp" else {}
            with tempfile.TemporaryDirectory() as tmp:
                fit = tr.take(~np.isin(tr.subject, test))
                bundle = engine.train(method, name, fit, config, kwargs, Path(tmp) / "fold.ckpt", log=None)
                engine.train(method, name, fit, config, kwargs, Path(tmp) / "fold.ckpt", log=None)   # resume path
            rows, ref, who = engine.evaluate(method, bundle, te.take(np.isin(te.subject, test)), 1,
                                             fit.bp.mean(axis=0), "cpu")
            assert len(ref) and all(np.all(np.isfinite(v)) and v.shape == ref.shape for v in rows.values())
            print(f"ok    {name:<7} {len(tr):5d} training windows, input {tr.x.shape[1:]}, {len(ref)} scored")
        except SystemExit as exc:                                    # a missing optional package
            print(f"skip  {name:<7} {exc}")
        except Exception as exc:
            failed.append(name)
            print(f"FAIL  {name:<7} {exc!r}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
