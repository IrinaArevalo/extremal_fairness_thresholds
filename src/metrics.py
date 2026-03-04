
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from config import MetricName
from typing import Optional, Tuple

def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den != 0 else np.nan


def _confusion_rates(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Return (TPR, FPR)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr = _safe_div(tp, tp + fn)
    fpr = _safe_div(fp, fp + tn)
    return float(tpr), float(fpr)


def fairness_metric_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    age: np.ndarray,
    t: float,
    metric: MetricName,
    decision_threshold: float = 0.5,
) -> Tuple[float, int, int]:
    """
    Compute disparity metric between groups:
      left  = age < t   (unprivileged, by convention)
      right = age >= t  (privileged)
    Returns (value, n_left, n_right).
    """
    left = age < t
    right = ~left
    nL, nR = int(left.sum()), int(right.sum())
    if nL == 0 or nR == 0:
        return np.nan, nL, nR

    yL, sL = y_true[left], y_score[left]
    yR, sR = y_true[right], y_score[right]
    yhatL = (sL >= decision_threshold).astype(int)
    yhatR = (sR >= decision_threshold).astype(int)

    if metric == "statistical_parity_difference":
        return float(np.mean(yhatL == 1) - np.mean(yhatR == 1)), nL, nR

    if metric in ("disparate_impact", "log_disparate_impact"):
        pL = float(np.mean(yhatL == 1))
        pR = float(np.mean(yhatR == 1))
        di = _safe_div(pL, pR)
        if metric == "disparate_impact":
            return float(di), nL, nR
        return float(np.log(di)) if di > 0 else np.nan, nL, nR

    if metric == "equal_opportunity_difference":
        tprL, _ = _confusion_rates(yL, yhatL)
        tprR, _ = _confusion_rates(yR, yhatR)
        return float(tprL - tprR), nL, nR

    if metric == "average_odds_difference":
        tprL, fprL = _confusion_rates(yL, yhatL)
        tprR, fprR = _confusion_rates(yR, yhatR)
        return float(0.5 * ((tprL - tprR) + (fprL - fprR))), nL, nR

    if metric == "auc_difference":
        aucL = roc_auc_score(yL, sL) if len(np.unique(yL)) == 2 else np.nan
        aucR = roc_auc_score(yR, sR) if len(np.unique(yR)) == 2 else np.nan
        return float(aucL - aucR), nL, nR

    raise ValueError(f"Unknown metric: {metric}")


def candidate_thresholds(age: np.ndarray, use_midpoints: bool = True, max_thresholds: Optional[int] = None) -> np.ndarray:
    a = np.asarray(age, dtype=float)
    a = a[~np.isnan(a)]
    u = np.unique(a)
    u.sort()
    if len(u) < 2:
        return u
    Ts = (u[:-1] + u[1:]) / 2.0 if use_midpoints else u
    if max_thresholds is not None and len(Ts) > max_thresholds:
        idx = np.linspace(0, len(Ts) - 1, num=max_thresholds).round().astype(int)
        Ts = Ts[idx]
    return Ts
