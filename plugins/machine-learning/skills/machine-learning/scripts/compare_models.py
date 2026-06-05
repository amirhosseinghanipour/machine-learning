#!/usr/bin/env python3
"""Compare two models' predictions with paired bootstrap CIs and a paired permutation test.

A single test-set number is an ESTIMATE with variance. Before claiming model A beats model B,
quantify the uncertainty and test whether the difference is real (not noise or selection). This
implements the two recommended tools from references/evaluation-statistics.md:

  1. Paired bootstrap  -> 95% CIs for each model's metric AND for their difference.
  2. Paired permutation test -> a p-value for "is A != B?" under the exchangeability null.

Both are PAIRED (the models are evaluated on the SAME examples), which is far more powerful than
comparing independent numbers. Pairing also correctly handles the correlation between the two
models' predictions on shared examples.

NOTE on scope: this captures TEST-SET sampling variance. For stochastic training, the dominant
variance is often BETWEEN training seeds -- run several seeds and combine both sources (see
references/evaluation-statistics.md section 5). This tool is also not a substitute for a fair,
equal-effort comparison (see SKILL.md): a significant win over an under-tuned baseline is still invalid.

Programmatic use:
    from compare_models import compare, accuracy
    res = compare(y_true, pred_a, pred_b, metric=accuracy, n_boot=10000, n_perm=10000, seed=0)
    print(res.report())

CLI (CSV or .npy columns of equal length; --metric accuracy by default):
    python compare_models.py --y-true labels.npy --pred-a a.npy --pred-b b.npy --metric accuracy
    # For pre-computed per-example scores (e.g. 0/1 correctness or per-example loss), use:
    python compare_models.py --scores-a a.csv --scores-b b.csv --metric mean --lower-is-better   # losses
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Optional

try:
    import numpy as np
except ImportError as e:  # numpy is required for this statistics utility
    raise SystemExit("compare_models.py requires numpy:  pip install numpy") from e


Metric = Callable[[np.ndarray, np.ndarray], float]


# --- a few common metrics; pass your own metric(y_true, y_pred) -> float for anything else ---
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """For pre-computed per-example scores: y_pred holds the per-example value, y_true is ignored."""
    return float(np.mean(y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


BUILTIN_METRICS = {"accuracy": accuracy, "mean": mean, "rmse": rmse, "mae": mae}


@dataclass
class CompareResult:
    metric_a: float
    metric_b: float
    diff: float  # metric_a - metric_b (on the full sample)
    ci_a: tuple[float, float]
    ci_b: tuple[float, float]
    ci_diff: tuple[float, float]
    p_permutation: float
    n: int
    n_boot: int
    n_perm: int
    lower_is_better: bool
    alpha: float

    @property
    def significant(self) -> bool:
        """Significant at alpha if the difference CI excludes 0 (and the permutation p agrees)."""
        ci_excludes_zero = not (self.ci_diff[0] <= 0.0 <= self.ci_diff[1])
        return ci_excludes_zero and self.p_permutation < self.alpha

    def report(self) -> str:
        better = "A" if (self.diff > 0) ^ self.lower_is_better else "B"
        direction = "lower=better" if self.lower_is_better else "higher=better"
        conf = int(round((1 - self.alpha) * 100))
        lines = [
            f"Paired comparison on n={self.n} examples ({direction}, "
            f"{self.n_boot} bootstraps, {self.n_perm} permutations)",
            f"  Model A: {self.metric_a:.4f}   {conf}% CI [{self.ci_a[0]:.4f}, {self.ci_a[1]:.4f}]",
            f"  Model B: {self.metric_b:.4f}   {conf}% CI [{self.ci_b[0]:.4f}, {self.ci_b[1]:.4f}]",
            f"  Diff (A - B): {self.diff:+.4f}   {conf}% CI "
            f"[{self.ci_diff[0]:+.4f}, {self.ci_diff[1]:+.4f}]",
            f"  Permutation p-value (two-sided): {self.p_permutation:.4g}",
            "",
            (
                f"  => Difference IS significant at alpha={self.alpha}: model {better} is better."
                if self.significant
                else f"  => NOT significant at alpha={self.alpha}: the difference is within noise. "
                "Do not claim a win."
            ),
        ]
        return "\n".join(lines)


def _percentile_ci(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    lo = float(np.percentile(samples, 100 * (alpha / 2)))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return lo, hi


def compare(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric: Metric = accuracy,
    n_boot: int = 10000,
    n_perm: int = 10000,
    alpha: float = 0.05,
    lower_is_better: bool = False,
    seed: Optional[int] = 0,
) -> CompareResult:
    """Paired bootstrap CIs + paired permutation test for metric(A) vs metric(B).

    y_true, pred_a, pred_b must be aligned, equal-length arrays over the SAME examples.
    For pre-computed per-example scores, pass metric=mean and put the scores in pred_a/pred_b
    (y_true is then ignored). `diff` is always metric_a - metric_b on the full sample.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    n = len(y_true)
    if not (len(pred_a) == len(pred_b) == n):
        raise ValueError(f"length mismatch: y_true={n}, pred_a={len(pred_a)}, pred_b={len(pred_b)}")
    if n == 0:
        raise ValueError("empty inputs")

    rng = np.random.default_rng(seed)

    metric_a = metric(y_true, pred_a)
    metric_b = metric(y_true, pred_b)
    diff = metric_a - metric_b

    # --- Paired bootstrap: resample EXAMPLE INDICES once per replicate, score both models on
    #     the same resample so the difference's CI accounts for their correlation. ---
    boot_a = np.empty(n_boot)
    boot_b = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_a[i] = metric(y_true[idx], pred_a[idx])
        boot_b[i] = metric(y_true[idx], pred_b[idx])
    boot_diff = boot_a - boot_b

    # --- Paired permutation test: under H0 the two prediction vectors are exchangeable, so for
    #     each example independently swap A<->B with prob 0.5 and rebuild the null of the diff. ---
    perm_diffs = np.empty(n_perm)
    for i in range(n_perm):
        swap = rng.random(n) < 0.5
        pa = np.where(swap, pred_b, pred_a)
        pb = np.where(swap, pred_a, pred_b)
        perm_diffs[i] = metric(y_true, pa) - metric(y_true, pb)
    # two-sided p with the conventional +1 correction (never reports p=0)
    p_perm = (np.sum(np.abs(perm_diffs) >= abs(diff)) + 1) / (n_perm + 1)

    return CompareResult(
        metric_a=metric_a,
        metric_b=metric_b,
        diff=diff,
        ci_a=_percentile_ci(boot_a, alpha),
        ci_b=_percentile_ci(boot_b, alpha),
        ci_diff=_percentile_ci(boot_diff, alpha),
        p_permutation=float(p_perm),
        n=n,
        n_boot=n_boot,
        n_perm=n_perm,
        lower_is_better=lower_is_better,
        alpha=alpha,
    )


def _load(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    # CSV / text: one value per line (or comma/whitespace separated, flattened)
    return np.loadtxt(path, delimiter="," if path.endswith(".csv") else None).ravel()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--y-true", help="labels/targets file (.npy/.csv); omit when --metric mean")
    p.add_argument("--pred-a", help="model A predictions file")
    p.add_argument("--pred-b", help="model B predictions file")
    p.add_argument("--scores-a", help="alias for --pred-a when passing per-example scores")
    p.add_argument("--scores-b", help="alias for --pred-b when passing per-example scores")
    p.add_argument("--metric", default="accuracy", choices=sorted(BUILTIN_METRICS), help="default: accuracy")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--lower-is-better", action="store_true", help="for losses/error metrics")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    pred_a = _load(args.pred_a or args.scores_a)
    pred_b = _load(args.pred_b or args.scores_b)
    y_true = _load(args.y_true) if args.y_true else np.zeros(len(pred_a))  # ignored by metric=mean

    res = compare(
        y_true,
        pred_a,
        pred_b,
        metric=BUILTIN_METRICS[args.metric],
        n_boot=args.n_boot,
        n_perm=args.n_perm,
        alpha=args.alpha,
        lower_is_better=args.lower_is_better,
        seed=args.seed,
    )
    print(res.report())


if __name__ == "__main__":
    main()
