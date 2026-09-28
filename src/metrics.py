"""
metrics.py - the official score: macro-averaged F0.5, computed per Source-1 entity.

For one S1 entity:  P = correct predicted / all predicted,  R = correct predicted / all true
                    F0.5 = 1.25*P*R / (0.25*P + R)
Special cases (from the problem statement):
  no true matches and nothing predicted  -> 1.0   (correct singleton)
  no true matches but something predicted -> 0.0
  true matches exist but nothing predicted -> 0.0
The final score is the plain average over all S1 entities.
"""
import numpy as np


def macro_f05(s1_ids, pred_pairs, true_pairs):
    """s1_ids: every S1 id being evaluated.
    pred_pairs / true_pairs: sets of (s1_id, other_id) tuples."""
    from collections import Counter
    n_pred = Counter(a for a, _ in pred_pairs)
    n_true = Counter(a for a, _ in true_pairs)
    tp = Counter(a for a, b in pred_pairs if (a, b) in true_pairs)
    scores = []
    for s in s1_ids:
        p, t, c = n_pred.get(s, 0), n_true.get(s, 0), tp.get(s, 0)
        if t == 0:
            scores.append(1.0 if p == 0 else 0.0)
        elif p == 0 or c == 0:
            scores.append(0.0)
        else:
            P, R = c / p, c / t
            scores.append(1.25 * P * R / (0.25 * P + R))
    return float(np.mean(scores)) if scores else 0.0


def macro_f05_arrays(n_s1, s1_idx, pred, label, n_true_per_s1):
    """Fast version on integer arrays.
    n_s1: number of S1 entities evaluated (indices 0..n_s1-1)
    s1_idx, pred (bool), label (bool): one entry per candidate pair
    n_true_per_s1: number of true matches of every S1 entity (including ones blocking missed)."""
    p = np.bincount(s1_idx, weights=pred, minlength=n_s1)
    c = np.bincount(s1_idx, weights=pred & label, minlength=n_s1)
    t = n_true_per_s1.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(p > 0, c / p, 0.0)
        R = np.where(t > 0, c / t, 0.0)
        f = np.where(c > 0, 1.25 * P * R / (0.25 * P + R), 0.0)
    f = np.where(t == 0, (p == 0).astype(float), f)
    return float(f.mean())
