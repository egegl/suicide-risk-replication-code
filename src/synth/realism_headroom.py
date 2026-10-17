"""Leak-free synthetic-vs-real classifier AUC (the pilot's "AUC" column).

The same matcher, vectorizer, and grouped five-fold CV as synth.realism, with
the vectorizer fit inside every training fold. Functions are copied unchanged
from the competition-era diagnostics module.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import realism


def case_records(cases: list[dict], arm: str = "synthetic") -> list[dict]:
    """Convert parsed/complete case rows into realism's lightweight records."""
    return [{
        "text": case["post"],
        "risk": case["risk"],
        "n_tokens": len(case["post"].split()),
        "user": f"{arm}:{case['case_id']}",
        "row_id": f"{arm}:{case['case_id']}",
    } for case in cases]


def _matched(records: list[dict], processed: Path) -> dict:
    tr = realism.train_pool(processed)
    reals, edges = realism.match_reals(records, tr)
    kept = [records[real["synth_idx"]] for real in reals]
    texts = [row["text"] for row in kept] + [row["text"] for row in reals]
    y = np.asarray([1] * len(kept) + [0] * len(reals))
    groups = [row["user"] for row in kept] + [row["user"] for row in reals]
    return {
        "synth": kept, "reals": reals, "texts": texts, "y": y,
        "groups": groups, "edges": edges,
    }


def _fit_lr(X, y):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1.0, max_iter=2000).fit(X, y)


def _folds(y, groups, n_splits: int = 5):
    """The shared fold map: deterministic in the group order, identical for
    every section that passes the same (y, groups)."""
    from sklearn.model_selection import GroupKFold
    dummy = np.zeros((len(y), 1))
    return list(GroupKFold(n_splits=n_splits).split(dummy, y, groups))


def _nested_cv(texts, y, groups, make_vec, n_splits: int = 5) -> dict:
    """The gate's CV loop, leak-free: featurizer fit on the train fold only.
    Fails loudly on a one-class test fold (the gate's loop skips it silently)."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    texts = np.asarray(texts, dtype=object)
    oof = np.full(len(y), np.nan)
    aucs, n_features = [], []
    for tr_i, te_i in _folds(y, groups, n_splits):
        assert len(set(y[te_i])) == 2, "one-class test fold: groups too skewed"
        vec = make_vec()
        Xtr = vec.fit_transform(list(texts[tr_i]))
        Xte = vec.transform(list(texts[te_i]))
        n_features.append(int(Xtr.shape[1]))
        p = _fit_lr(Xtr, y[tr_i]).predict_proba(Xte)[:, 1]
        oof[te_i] = p
        aucs.append(float(roc_auc_score(y[te_i], p)))
    return {"auc_per_fold": [round(a, 4) for a in aucs],
            "auc_mean": round(float(np.mean(aucs)), 4),
            "n_features_per_fold": n_features, "oof": oof}


def leakfree_auc(records: list[dict], processed: Path) -> dict:
    """Composite detector with a fresh vectorizer fit inside every fold."""
    matched = _matched(records, processed)
    result = _nested_cv(
        matched["texts"], matched["y"], matched["groups"],
        realism.default_vectorizer)
    result.pop("oof", None)
    result.update({
        "n_synthetic": len(matched["synth"]),
        "n_real": len(matched["reals"]),
        "vectorizer_scope": "grouped-CV training fold only",
    })
    return result
