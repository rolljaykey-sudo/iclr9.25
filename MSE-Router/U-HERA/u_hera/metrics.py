"""Unrounded MOSI metrics, compatible with the existing Has0/Non0 definitions."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def sentiment_metrics(labels, predictions):
    y, p = np.asarray(labels, dtype=np.float64), np.asarray(predictions, dtype=np.float64)
    if y.shape != p.shape or not len(y) or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("invalid prediction/label arrays")
    result = dict(MAE=float(np.abs(p-y).mean()),
                  Corr=float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 and np.std(y) > 0 else None,
                  Mult_acc_7=float(np.mean(np.round(np.clip(p, -3, 3)) == np.round(np.clip(y, -3, 3)))))
    for name, keep, truth, pred in (
        ("Has0", np.ones(len(y), dtype=bool), y >= 0, p >= 0),
        ("Non0", y != 0, y > 0, p > 0),
    ):
        result[name + "_acc_2"] = float(accuracy_score(truth[keep], pred[keep])) if keep.any() else None
        result[name + "_F1_score"] = float(f1_score(truth[keep], pred[keep], average="weighted", zero_division=0)) if keep.any() else None
    return result


def metrics_from_rows(rows):
    result = sentiment_metrics([r["label"] for r in rows], [r["prediction"] for r in rows])
    result["invalid_generation_rate"] = sum(r["invalid"] or r["out_of_range"] for r in rows) / len(rows)
    result["unparseable_rate"] = sum(r["invalid"] for r in rows) / len(rows)
    result["out_of_range_rate"] = sum(r["out_of_range"] for r in rows) / len(rows)
    if "G" in rows[0]:
        result["mean_G_T_A_V_Null"] = np.asarray([r["G"] for r in rows]).mean(0).tolist()
    result["samples"] = len(rows)
    return result
