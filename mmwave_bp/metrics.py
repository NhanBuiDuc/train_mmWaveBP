"""Accuracy metrics and validation criteria for BP estimates (pure numpy/scipy).

Error convention: e = estimate - reference (ISO 81060-2). Criteria:
* AAMI / ISO 81060-2 criterion 1: |ME| <= 5 mmHg and SD <= 8 mmHg over all pairs.
* ISO 81060-2 criterion 2: SD of per-subject mean errors <= SD_max(|ME|), where SD_max solves
  P(|e| <= 10) = 0.85 for a normal error (reproduces the published table: 6.95 at 0, 4.81 at 5).
* BHS grade from cumulative % within 5/10/15 mmHg; IEEE 1708 grade from MAD.
* PTE (probability of tolerable error, Yu & Lowe): P(|e| <= 10) with e | BP linear-normal,
  integrated over the standard's BP distribution (SBP N(130, 20), DBP N(80, 13)); passes at 0.85.
* BP change (IEEE 1708a, ISO 81060-3, Lancet review): errors only on readings that moved
  >= 10 mmHg from the subject's calibration mean, and whether the estimate moved the same way.
  A cuffless method that cannot follow changes is no better than repeating the calibration value.
Criteria are reported as not applicable when fewer than 90 % of the readings have an estimate.
With far fewer than 85 subjects these are feasibility numbers, never a compliance claim.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats

BHS_GRADES = (("A", (60, 85, 95)), ("B", (50, 75, 90)), ("C", (40, 65, 85)))
STANDARD_BP = {"sbp": (130.0, 20.0), "dbp": (80.0, 13.0)}
PTE_PASS = 0.85            # ISO 81060-2: P(|e| <= 10 mmHg) >= 0.85
MIN_COVERAGE = 0.9         # below this the criteria are not applicable
MIN_CHANGE_MMHG = 10.0     # a "BP change" reading (IEEE 1708a uses induced changes of ~10-30 mmHg)


def sd_max(mean_error: float) -> float:
    """ISO 81060-2 criterion 2 limit for a given mean error (NaN when |ME| > 5: fail)."""
    me = abs(mean_error)
    if me > 5:
        return float("nan")
    return float(optimize.brentq(lambda s: stats.norm.cdf((10 - me) / s) - stats.norm.cdf((-10 - me) / s) - 0.85,
                                 0.1, 50))


def bhs_grade(abs_error: np.ndarray) -> tuple[str, tuple[float, float, float]]:
    within = tuple(float(np.mean(abs_error <= t) * 100) for t in (5, 10, 15))
    for grade, limits in BHS_GRADES:
        if all(w >= l for w, l in zip(within, limits)):
            return grade, within
    return "D", within


def ieee1708_grade(mad: float) -> str:
    return "A" if mad <= 5 else "B" if mad <= 6 else "C" if mad <= 7 else "D"


def pte(estimate: np.ndarray, reference: np.ndarray, target: str) -> float:
    """Probability of |error| <= 10 mmHg over the standard population distribution."""
    ref, err = np.asarray(reference, float), np.asarray(estimate, float) - np.asarray(reference, float)
    if len(ref) < 3:
        return float("nan")
    slope, intercept = np.polyfit(ref, err, 1) if np.ptp(ref) > 0 else (0.0, err.mean())
    sigma = np.std(err - (intercept + slope * ref), ddof=2) + 1e-9
    mu, sd = STANDARD_BP[target]
    bp = np.linspace(mu - 4 * sd, mu + 4 * sd, 801)
    bias = intercept + slope * bp
    p_ok = stats.norm.cdf((10 - bias) / sigma) - stats.norm.cdf((-10 - bias) / sigma)
    weight = stats.norm.pdf(bp, mu, sd)
    return float(np.sum(p_ok * weight) / np.sum(weight))


def within_subject_tracking(estimate: np.ndarray, reference: np.ndarray, subjects: np.ndarray) -> float:
    """Correlation of changes: both centred on each subject's mean (NaN if no variation)."""
    est, ref = np.asarray(estimate, float).copy(), np.asarray(reference, float).copy()
    for s in np.unique(subjects):
        m = subjects == s
        est[m] -= est[m].mean()
        ref[m] -= ref[m].mean()
    if np.std(est) < 1e-9 or np.std(ref) < 1e-9:
        return float("nan")
    return float(np.corrcoef(est, ref)[0, 1])


def change_metrics(estimate: np.ndarray, reference: np.ndarray, calibration: np.ndarray,
                   min_change: float = MIN_CHANGE_MMHG) -> dict:
    """Readings whose reference moved >= `min_change` from the subject's calibration mean
    (`calibration`, one value per reading): error there, and direction concordance
    P(sign(est - cal) == sign(ref - cal)). Carrying the calibration value forward scores 0.5-."""
    est, ref, cal = (np.asarray(v, float) for v in (estimate, reference, calibration))
    ok = np.isfinite(est) & np.isfinite(ref) & np.isfinite(cal)
    change = ref - cal
    big = ok & (np.abs(change) >= min_change)
    out = {"readings": int(big.sum()), "changes_of_all": float(big.sum() / max(ok.sum(), 1)),
           "median_abs_change": float(np.median(np.abs(change[ok]))) if ok.any() else float("nan")}
    if big.sum() >= 2:
        err = est[big] - ref[big]
        out.update({"me": float(err.mean()), "sd": float(err.std(ddof=1)), "mae": float(np.abs(err).mean()),
                    "direction_agreement": float(np.mean(np.sign(est[big] - cal[big]) == np.sign(change[big])))})
    return out


def metrics(estimate: np.ndarray, reference: np.ndarray, subjects: np.ndarray, target: str,
            calibration: np.ndarray | None = None) -> dict:
    """Full metric set for one target (sbp or dbp). NaN estimates count against coverage.
    `calibration` (the subject's calibration mean per reading) adds the BP-change metrics.
    Criterion 2 uses the mean of per-subject mean errors (equal to ME when subjects have equal
    numbers of readings, as the standard's protocol has)."""
    est, ref, subjects = np.asarray(estimate, float), np.asarray(reference, float), np.asarray(subjects)
    valid = np.isfinite(est) & np.isfinite(ref)
    out = {"target": target, "pairs": int(len(ref)), "coverage": float(valid.mean()) if len(ref) else 0.0}
    if calibration is not None:
        out["bp_change"] = change_metrics(est, ref, calibration)
    est, ref, subjects = est[valid], ref[valid], subjects[valid]
    if len(ref) < 2:
        return out
    applicable = out["coverage"] >= MIN_COVERAGE
    err = est - ref
    me, sd = float(err.mean()), float(err.std(ddof=1))
    grade, within = bhs_grade(np.abs(err))
    subject_means = np.array([err[subjects == s].mean() for s in np.unique(subjects)])
    me2 = float(subject_means.mean())
    sd2 = float(subject_means.std(ddof=1)) if len(subject_means) > 1 else float("nan")
    limit = sd_max(me2)
    slope = float(np.polyfit(ref, err, 1)[0]) if np.ptp(ref) > 0 else float("nan")
    out.update({
        "subjects": int(len(np.unique(subjects))), "reference_mean": float(ref.mean()),
        "reference_sd": float(ref.std(ddof=1)), "me": me, "sd": sd, "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())), "within_5_10_15": within, "bhs": grade,
        "ieee1708": ieee1708_grade(float(np.abs(err).mean())),
        "criteria_applicable": applicable,
        "aami_criterion1": bool(applicable and abs(me) <= 5 and sd <= 8),
        "iso_criterion2": {"me": me2, "sd": sd2, "sd_max": limit,
                           "pass": bool(applicable and np.isfinite(limit) and np.isfinite(sd2) and sd2 <= limit)},
        "pte": (p := pte(est, ref, target)), "pte_pass": bool(applicable and p >= PTE_PASS), "r": float(np.corrcoef(est, ref)[0, 1]) if np.std(est) > 0 else float("nan"),
        "sd_ratio": float(np.std(est) / (np.std(ref) + 1e-9)), "error_vs_reference_slope": slope,
        "tracking_r": within_subject_tracking(est, ref, subjects),
    })
    return out


def skill(estimate: np.ndarray, baseline: np.ndarray, reference: np.ndarray) -> float:
    """1 - MAE_model / MAE_baseline on the readings both have: <= 0 = nothing beyond the baseline."""
    est, base, ref = (np.asarray(v, float) for v in (estimate, baseline, reference))
    ok = np.isfinite(est) & np.isfinite(base) & np.isfinite(ref)
    if not ok.any():
        return float("nan")
    base_mae = np.abs(base[ok] - ref[ok]).mean()
    return float(1 - np.abs(est[ok] - ref[ok]).mean() / base_mae) if base_mae > 0 else float("nan")
