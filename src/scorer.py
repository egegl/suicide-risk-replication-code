import json
import re
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score
from prep import constants as C

def norm(s: str) -> str:
    return re.sub('\\s+', ' ', s.lower().strip())

def toklen(s: str) -> int:
    return len(norm(s).split())

def match(pred: str, gold: str) -> bool:
    p, g = (norm(pred), norm(gold))
    return (p in g or g in p) and toklen(pred) <= 3 * toklen(gold)

def phrase_f1_post(preds: list[str], golds: list[str]) -> float:
    preds = [p for p in preds if norm(p)]
    golds = [g for g in golds if norm(g)]
    if not golds and (not preds):
        return 1.0
    if not golds or not preds:
        return 0.0
    cost = np.array([[-int(match(p, g)) for g in golds] for p in preds])
    ri, ci = linear_sum_assignment(cost)
    m = int(-cost[ri, ci].sum())
    P, R = (m / len(preds), m / len(golds))
    return 2 * P * R / (P + R) if P + R else 0.0

def phrase_f1(all_preds: list[list[str]], all_golds: list[list[str]], empty_convention: str='ee1') -> float:
    assert len(all_preds) == len(all_golds)
    assert empty_convention in ('ee1', 'skip'), empty_convention
    pairs = list(zip(all_preds, all_golds))
    if empty_convention == 'skip':
        pairs = [(p, g) for p, g in pairs if any((norm(x) for x in g))]
        if not pairs:
            return 0.0
    return float(np.mean([phrase_f1_post(p, g) for p, g in pairs]))

def risk_wf1(y_true: list[str], y_pred: list[str]) -> float:
    return float(f1_score(y_true, y_pred, labels=list(C.RISK_CLASSES), average='weighted', zero_division=0))

def factors_macro_f1(true_sets: list[list[str]], pred_sets: list[list[str]]) -> float:

    def mat(sets):
        m = np.zeros((len(sets), len(C.FACTORS_24)), dtype=np.uint8)
        for i, lst in enumerate(sets):
            for f in lst:
                m[i, C.FACTOR_INDEX[f]] = 1
        return m
    return float(f1_score(mat(true_sets), mat(pred_sets), average='macro', zero_division=0))

def load_gold(path: str | Path, fold: int | None=None) -> dict[str, dict]:
    gold = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if fold is None or r['fold'] == fold:
                gold[r['row_id']] = r
    return gold

def score_slice(preds: dict[str, dict], gold: dict[str, dict], ids, empty_convention: str='ee1') -> dict[str, float]:
    sub_gold = {i: gold[i] for i in ids if i in gold}
    return score({i: preds[i] for i in sub_gold}, sub_gold, empty_convention=empty_convention)

def hard_slice_report(preds: dict[str, dict], gold: dict[str, dict], hard_slice_path: str | Path) -> dict:
    hs = json.loads(Path(hard_slice_path).read_text())
    by_tag: dict[str, list[str]] = {}
    for rid, tag in hs['curated'].items():
        if rid in gold:
            by_tag.setdefault(tag, []).append(rid)
    out = {tag: {'n': len(ids), 'risk_acc': round(sum((preds[i]['risk'] == gold[i]['risk'] for i in ids)) / len(ids), 3), 'miss_ids': sorted((i for i in ids if preds[i]['risk'] != gold[i]['risk']))} for tag, ids in sorted(by_tag.items())}
    out['_curated_overall'] = {k: round(v, 4) for k, v in score_slice(preds, gold, list(hs['curated'])).items()}
    return out

def score(preds: dict[str, dict], gold: dict[str, dict], empty_convention: str='ee1') -> dict[str, float]:
    missing = set(gold) - set(preds)
    assert not missing, f'predictions missing for {len(missing)} rows, e.g. {sorted(missing)[:5]}'
    ids = sorted(gold)
    w = risk_wf1([gold[i]['risk'] for i in ids], [preds[i]['risk'] for i in ids])
    p = phrase_f1([preds[i]['spans'] for i in ids], [gold[i]['spans'] for i in ids], empty_convention=empty_convention)
    m = factors_macro_f1([gold[i]['factors'] for i in ids], [preds[i]['factors'] for i in ids])
    composite = 0.4 * w + 0.3 * p + 0.3 * m
    return {'risk_wf1': w, 'phrase_f1': p, 'macro_f1': m, 'composite': composite, 's1': (0.4 * w + 0.3 * p) / 0.7, 's2': m, 'n_posts': len(ids)}
