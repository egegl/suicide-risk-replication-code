from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
import scorer
from prep import constants as PC
from synth import constants as SC
from . import constants as C
from . import oof
SYNTHETIC_ARMS = ('synthetic',)

def rare_class_f1(preds: dict[str, dict], gold: dict[str, dict]) -> dict[str, float]:
    missing = [rid for rid in gold if rid not in preds]
    assert not missing, f'rare_class_f1: preds miss {len(missing)} gold rows (unpropagated pooled OOF? run oof.propagate first), e.g. {sorted(missing)[:5]}'
    out = {}
    for f in SC.RARE_FACTORS:
        tp = fp = fn = 0
        for rid, g in gold.items():
            t = f in g['factors']
            p = f in preds[rid]['factors']
            tp += t and p
            fp += p and (not t)
            fn += t and (not p)
        denom = 2 * tp + fp + fn
        out[f] = round(2 * tp / denom if denom else 0.0, 4)
    return out

def _pred_arrays(rows_full: dict[str, dict], gold: dict[str, dict], ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    risk = np.array([rows_full[i]['risk'] for i in ids], dtype=object)
    phrase = np.array([scorer.phrase_f1_post(rows_full[i]['spans'], gold[i]['spans']) for i in ids], dtype=float)
    fmat = np.zeros((len(ids), len(PC.FACTORS_24)), dtype=np.uint8)
    for r, i in enumerate(ids):
        for f in rows_full[i]['factors']:
            fmat[r, PC.FACTOR_INDEX[f]] = 1
    return (risk, phrase, fmat)

def _as_preds(rows_or_preds) -> dict[str, dict]:
    if isinstance(rows_or_preds, dict):
        return {rid: {'risk': p['risk'], 'spans': list(p['spans']), 'factors': list(p['factors'])} for rid, p in rows_or_preds.items()}
    return {r['row_id']: {'risk': r['risk'], 'spans': list(r['spans']), 'factors': list(r['factors'])} for r in rows_or_preds}

def paired_bootstrap(per_run_rows: dict, gold: dict[str, dict], users_by_row: dict[str, str], processed: Path | None=None, b: int=C.BOOTSTRAP_B, seed: int=C.BOOTSTRAP_SEED, convention: str=C.EE_PRIMARY, metric: str='composite') -> dict:
    assert 'real_only' in per_run_rows, "paired_bootstrap needs a 'real_only' label"
    assert convention in ('ee1', 'skip'), convention
    assert metric in ('composite', 'rare_macro'), metric
    ids = sorted(gold)
    missing_users = [i for i in ids if i not in users_by_row]
    assert not missing_users, f'users_by_row misses gold rows, e.g. {missing_users[:5]}'
    risk_true = np.array([gold[i]['risk'] for i in ids], dtype=object)
    gold_fmat = np.zeros((len(ids), len(PC.FACTORS_24)), dtype=np.uint8)
    for r, i in enumerate(ids):
        for f in gold[i]['factors']:
            gold_fmat[r, PC.FACTOR_INDEX[f]] = 1
    gold_nonempty = np.array([any((scorer.norm(x) for x in gold[i]['spans'])) for i in ids], dtype=bool)
    label_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for label, rows in per_run_rows.items():
        preds = _as_preds(rows)
        full = oof.propagate(preds, Path(processed)) if processed is not None else preds
        missing = [i for i in ids if i not in full]
        assert not missing, f'{label}: preds miss gold rows, e.g. {missing[:5]}'
        label_arrays[label] = _pred_arrays(full, gold, ids)
    users = sorted({users_by_row[i] for i in ids})
    row_idx_by_user = {u: [] for u in users}
    for r, i in enumerate(ids):
        row_idx_by_user[users_by_row[i]].append(r)
    user_rows = [np.array(row_idx_by_user[u], dtype=np.intp) for u in users]
    rare_cols = np.array([PC.FACTOR_INDEX[f] for f in SC.RARE_FACTORS], dtype=np.intp)

    def composite(idx: np.ndarray, arrays) -> float:
        risk_pred, phrase, fmat = arrays
        if metric == 'rare_macro':
            return float(f1_score(gold_fmat[idx][:, rare_cols], fmat[idx][:, rare_cols], average='macro', zero_division=0))
        w = f1_score(list(risk_true[idx]), list(risk_pred[idx]), labels=list(PC.RISK_CLASSES), average='weighted', zero_division=0)
        if convention == 'skip':
            keep = gold_nonempty[idx]
            p = float(phrase[idx][keep].mean()) if keep.any() else 0.0
        else:
            p = float(phrase[idx].mean())
        m = f1_score(gold_fmat[idx], fmat[idx], average='macro', zero_division=0)
        return 0.4 * float(w) + 0.3 * p + 0.3 * float(m)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(users), size=(b, len(users)))
    idx_all = np.arange(len(ids), dtype=np.intp)
    reps = {label: np.empty(b, dtype=float) for label in per_run_rows}
    for rep_i in range(b):
        idx = np.concatenate([user_rows[j] for j in draws[rep_i]])
        for label, arrays in label_arrays.items():
            reps[label][rep_i] = composite(idx, arrays)
    out_labels = {}
    for label, arrays in label_arrays.items():
        out_labels[label] = {'observed': composite(idx_all, arrays), 'mean': float(reps[label].mean()), 'se': float(reps[label].std(ddof=1)), 'replicates': [float(x) for x in reps[label]]}
    deltas = {}
    for label in per_run_rows:
        if label == 'real_only':
            continue
        d = reps[label] - reps['real_only']
        deltas[label] = {'mean': float(d.mean()), 'se': float(d.std(ddof=1)), 'ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return {'unit': 'anon_user_id', 'n_users': len(users), 'n_rows': len(ids), 'b': b, 'seed': seed, 'convention': convention, 'metric': metric, 'labels': out_labels, 'deltas': deltas}

def one_se_ship(table: dict) -> dict:
    labels = table['labels']
    assert 'real_only' in labels, 'one_se_ship needs a real_only label'
    reps = {l: np.asarray(v['replicates'], dtype=float) for l, v in labels.items()}
    synthetic_arms = [a for a in SYNTHETIC_ARMS if a in labels]
    assert synthetic_arms, f'no synthetic arm among {sorted(labels)}'
    means = {l: labels[l]['mean'] for l in labels}
    best = max(synthetic_arms, key=lambda a: means[a])
    candidate, one_se = (best, 0.0)
    for arm in synthetic_arms:
        se_pair = float((reps[best] - reps[arm]).std(ddof=1))
        if means[arm] >= means[best] - se_pair:
            candidate, one_se = (arm, se_pair)
            break
    d_base = reps[candidate] - reps['real_only']
    beats_real_only = float(d_base.mean()) >= C.SHIP_BAR
    if 'repeated_real' in reps:
        d_q = reps[candidate] - reps['repeated_real']
        delta_vs_repeated_real = float(d_q.mean())
        beats_repeated_real = delta_vs_repeated_real > 0.0
    else:
        delta_vs_repeated_real = None
        beats_repeated_real = False
    return {'best': best, 'candidate': candidate, 'one_se': one_se, 'ship': bool(beats_real_only and beats_repeated_real), 'ship_bar': C.SHIP_BAR, 'delta_vs_real_only': float(d_base.mean()), 'beats_real_only': beats_real_only, 'delta_vs_repeated_real': delta_vs_repeated_real, 'beats_repeated_real': beats_repeated_real, 'repeated_real_present': 'repeated_real' in reps, 'convention': table['convention']}

def utility_gate(rare_boot: dict, decision: dict) -> dict:
    assert rare_boot.get('metric') == 'rare_macro', f"utility_gate: rare_boot carries metric={rare_boot.get('metric')!r} — pass the rare_macro table, not the composite one"
    cand = decision['candidate']
    reps = {l: np.asarray(v['replicates'], dtype=float) for l, v in rare_boot['labels'].items()}
    assert 'real_only' in reps, "utility_gate: rare_boot lacks a 'real_only' label"
    assert cand in reps, f'utility_gate: candidate {cand!r} not in rare_boot labels {sorted(reps)} — decision and rare table cover different arm sets'
    d_base = reps[cand] - reps['real_only']
    if 'repeated_real' in reps:
        d_q = reps[cand] - reps['repeated_real']
        rare_delta_q = float(d_q.mean())
        rare_q_ci = [float(np.percentile(d_q, 2.5)), float(np.percentile(d_q, 97.5))]
        rare_beats_q = rare_delta_q > 0.0
    else:
        rare_delta_q, rare_q_ci = (None, None)
        rare_beats_q = False
    return {'gate': 'utility', 'pass': bool(decision['ship'] and rare_beats_q), 'candidate': cand, 'composite': decision, 'rare_macro': {l: {k: v[k] for k in ('observed', 'mean', 'se')} for l, v in rare_boot['labels'].items()}, 'rare_delta_vs_real_only': {'mean': float(d_base.mean()), 'ci95': [float(np.percentile(d_base, 2.5)), float(np.percentile(d_base, 97.5))]}, 'rare_delta_vs_repeated_real': {'mean': rare_delta_q, 'ci95': rare_q_ci}, 'rare_beats_repeated_real': rare_beats_q, 'repeated_real_present': 'repeated_real' in reps}

def sweep_table(arm_summaries: dict[str, dict], bootstrap: dict, decision: dict) -> dict:
    arms = {}
    for arm in C.ARMS:
        if arm not in arm_summaries:
            continue
        entry = dict(arm_summaries[arm])
        if arm in bootstrap['labels']:
            lab = bootstrap['labels'][arm]
            entry['bootstrap'] = {k: lab[k] for k in ('observed', 'mean', 'se')}
        if arm in bootstrap['deltas']:
            entry['delta_vs_real_only'] = bootstrap['deltas'][arm]
        arms[arm] = entry
    return {'arms': arms, 'bootstrap': {k: bootstrap[k] for k in ('unit', 'n_users', 'n_rows', 'b', 'seed', 'convention')}, 'decision': decision, 'primary_convention': C.EE_PRIMARY, 'ship_bar': C.SHIP_BAR}
