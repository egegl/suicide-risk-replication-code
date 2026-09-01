import numpy as np
from prep import constants as PC
from . import constants as C

def _factors_of(g) -> list[str]:
    return g['factors'] if isinstance(g, dict) else g

def _argmax_tau(p: np.ndarray, y: np.ndarray) -> float:
    uniq = np.unique(p)
    cands = np.unique(np.concatenate([(uniq[:-1] + uniq[1:]) / 2.0, np.array([0.0, 1.0])]))
    pred = p[None, :] >= cands[:, None]
    tp = (pred & y).sum(axis=1)
    fp = (pred & ~y).sum(axis=1)
    fn = int(y.sum()) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    best_idx = int(np.nonzero(f1 == f1.max())[0][-1])
    return float(cands[best_idx])

def _tuning_arrays(rows: list[dict], gold_factors: dict, exclude_fold):
    pool = [r for r in rows if r['fold'] != exclude_fold]
    assert pool, f'no tuning rows left after excluding fold {exclude_fold}'
    gold_sets = {r['row_id']: frozenset(_factors_of(gold_factors[r['row_id']])) for r in pool}
    out = {}
    for factor in PC.FACTORS_24:
        p = np.array([r['p_true'][factor] for r in pool], dtype=float)
        y = np.array([factor in gold_sets[r['row_id']] for r in pool], dtype=bool)
        out[factor] = (p, y)
    return out

def fit_thresholds(rows: list[dict], gold_factors: dict, exclude_fold: int) -> dict[str, float]:
    arrays = _tuning_arrays(rows, gold_factors, exclude_fold)
    return {factor: C.THRESH_FALLBACK if not y.any() else _argmax_tau(p, y) for factor, (p, y) in arrays.items()}

def fit_thresholds_bagged(rows: list[dict], gold_factors: dict, exclude_fold: int, n_boot: int=C.THRESH_BAG_N_BOOT, seed: int=C.THRESH_BAG_SEED) -> dict[str, float]:
    arrays = _tuning_arrays(rows, gold_factors, exclude_fold)
    fold_key = 0 if exclude_fold is None else int(exclude_fold) + 1
    taus: dict[str, float] = {}
    for factor, (p, y) in arrays.items():
        if not y.any():
            taus[factor] = C.THRESH_FALLBACK
            continue
        rng = np.random.default_rng([seed, fold_key, PC.FACTOR_INDEX[factor]])
        n = len(p)
        bag = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            yb = y[idx]
            if yb.any():
                bag.append(_argmax_tau(p[idx], yb))
        taus[factor] = float(np.mean(bag)) if bag else _argmax_tau(p, y)
    return taus
FITTERS = {'argmax': fit_thresholds, 'bagged': fit_thresholds_bagged}

def apply_thresholds(rows: list[dict], taus_by_fold: dict[int, dict[str, float]], gate=None, rare_rules=None) -> list[dict]:
    assert gate is None and rare_rules is None, 'gate/rare_rules are hooks; not implemented'
    out = []
    for r in rows:
        taus = taus_by_fold[r['fold']]
        new = dict(r)
        new['factors'] = [f for f in PC.FACTORS_24 if r['p_true'][f] >= taus[f]]
        out.append(new)
    return out

def nested_thresholded_pool(pooled_rows: list[dict], gold_factors: dict[str, list[str]], estimator: str='argmax') -> tuple[list[dict], dict[int, dict[str, float]]]:
    fit = FITTERS[estimator]
    folds = sorted({r['fold'] for r in pooled_rows})
    taus_by_fold = {k: fit(pooled_rows, gold_factors, exclude_fold=k) for k in folds}
    return (apply_thresholds(pooled_rows, taus_by_fold), taus_by_fold)
