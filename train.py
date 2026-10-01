"""Train one method on one data set with subject-wise cross-validation.

    python train.py --method wavebp --dataset erlangen --data data/erlangen --out runs/wavebp
    python train.py --method rfbp   --dataset erlangen --data data/erlangen --out runs/rfbp --epochs 40
    python train.py --method airbp  --dataset blumio   --data data/blumio   --out runs/airbp --pretrain-epochs 30
    python train.py --method mmbp   --dataset blumio   --data data/blumio   --out runs/mmbp
    python train.py --method hbpfi  --dataset npz      --data data/my_arm   --out runs/hbpfi

Every fold trains on some subjects and is scored on the others. A run can be interrupted and
started again with the same command: finished folds are kept, the current fold continues from
its last finished epoch. `--final` then trains one model on all subjects (out/model.pt).
The printed table and out/report.json compare the model with two baselines; read them before
using a model (engine.py explains the rows).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mmwave_bp import data, engine, methods                         # noqa: E402


def parse_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=methods.NAMES)
    ap.add_argument("--dataset", required=True, choices=sorted(data.LOADERS))
    ap.add_argument("--data", type=Path, required=True, help="folder with the extracted data set")
    ap.add_argument("--out", type=Path, required=True, help="run folder (checkpoints, report, model)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-folds", type=int, default=None, help="stop after this many folds")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--pretrain-epochs", type=int, default=0, help="airBP: masked auto-encoder epochs")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--flip", action="store_true", help="random sign flip of every training window")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--cal", type=int, default=3, help="calibration windows per test subject (never scored)")
    ap.add_argument("--train-step", type=float, default=None, help="seconds between training windows")
    ap.add_argument("--max-subjects", type=int, default=None, help="use only the first N subjects (quick test)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="model arguments, e.g. d_model=256 layers=12")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--final", action="store_true", help="also train one model on all subjects")
    ap.add_argument("--fresh", action="store_true", help="ignore checkpoints and cached windows of this run")
    args = ap.parse_args()

    method = methods.get(args.method)
    model_kwargs = {k: parse_value(v) for k, v in (item.split("=", 1) for item in args.set)}
    config = engine.TrainConfig(args.epochs, args.pretrain_epochs, args.lr, args.batch, args.seed, args.flip,
                                args.val_fraction, args.patience, args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        for f in [*args.out.glob("fold*.ckpt"), *args.out.glob("fold*.npz"), *args.out.glob("windows_*.npz"),
                  *args.out.glob("final.ckpt")]:
            f.unlink()

    print(f"loading {args.dataset} from {args.data} ...")
    recordings = data.LOADERS[args.dataset](args.data)
    if args.max_subjects:
        keep = sorted({r.subject for r in recordings})[: args.max_subjects]
        recordings = [r for r in recordings if r.subject in keep]
    if not recordings:
        raise SystemExit(f"no recording found under {args.data}")
    hours = sum(r.n for r in recordings) / data.FS / 3600
    print(f"{len(recordings)} recordings, {len({r.subject for r in recordings})} subjects, {hours:.1f} h")

    train_step = args.train_step or method.TRAIN_STEP_S
    tag = f"{args.dataset}_{len(recordings)}"
    tr_all = engine.make_examples(method, recordings, train_step, args.out / f"windows_train_{tag}_{train_step:g}.npz")
    te_all = engine.make_examples(method, recordings, method.TEST_STEP_S, args.out / f"windows_test_{tag}.npz")
    print(f"{len(tr_all)} training windows, {len(te_all)} test windows; "
          f"mean BP {tr_all.bp[:, 0].mean():.1f} +- {tr_all.bp[:, 0].std():.1f} / "
          f"{tr_all.bp[:, 1].mean():.1f} +- {tr_all.bp[:, 1].std():.1f} mmHg")

    folds = data.subject_folds(tr_all.subject, args.folds, args.seed)
    settings = {"method": args.method, "dataset": args.dataset, "folds": folds, "cal": args.cal,
                "train_step": train_step, "model": model_kwargs,
                "config": {k: v for k, v in vars(config).items() if k != "device"}}
    (args.out / "run.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    rows, ref, who = {}, [], []
    for f, test_subjects in enumerate(folds[: args.max_folds], 1):
        done = args.out / f"fold{f}.npz"
        if not done.exists():
            print(f"fold {f}/{len(folds)}: {len(test_subjects)} held-out subjects")
            tr = tr_all.take(~np.isin(tr_all.subject, test_subjects))
            te = te_all.take(np.isin(te_all.subject, test_subjects))
            bundle = engine.train(method, args.method, tr, config, model_kwargs, args.out / f"fold{f}.ckpt")
            pred, r, s = engine.evaluate(method, bundle, te, args.cal, tr.bp.mean(axis=0), args.device)
            np.savez(done, reference=r, subjects=s, **pred)
        d = np.load(done, allow_pickle=False)
        for k in d.files:
            if k not in ("reference", "subjects"):
                rows.setdefault(k, []).append(d[k])
        ref.append(d["reference"]); who.append(d["subjects"])
    ref, who = np.concatenate(ref), np.concatenate(who)
    print(f"\n{args.method} on {args.dataset}: {len(ref)} test windows of {len(set(who))} unseen subjects")
    result = engine.report({k: np.concatenate(v) for k, v in rows.items()}, ref, who)
    (args.out / "report.json").write_text(json.dumps({**settings, "test_windows": int(len(ref)), "rows": result},
                                                     indent=2, default=float), encoding="utf-8")

    if args.final:
        print("\nfinal model on all subjects ...")
        bundle = engine.train(method, args.method, tr_all, config, model_kwargs, args.out / "final.ckpt")
        bundle.update({"trained_on": args.dataset, "cross_validation": result})
        engine.save_bundle(bundle, args.out / ("model.pkl" if bundle["kind"] == "sklearn" else "model.pt"))
        print(f"saved in {args.out}: its expected error on a new person is the table above, "
              "not its training loss")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
