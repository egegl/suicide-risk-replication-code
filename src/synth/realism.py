import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import constants as SC
from . import state

def _length_decile(n_tokens: int, edges: list[float]) -> int:
    for i, e in enumerate(edges):
        if n_tokens <= e:
            return i
    return len(edges)

def _collect_synth(pool_records: dict, prov_by_id: dict) -> list[dict]:
    out = []
    for pool in SC.POOLS:
        for r in pool_records.get(pool, []):
            prov = prov_by_id[r['row_id']]
            text = prov['timeline_texts'][prov['focal_pos']]
            risk = json.loads(r['target'])['risk']
            out.append({'text': text, 'risk': risk, 'n_tokens': len(text.split()), 'user': r['synth']['su_id'], 'row_id': r['row_id']})
    return out

def train_pool(processed: Path) -> pd.DataFrame:
    posts = pd.read_parquet(processed / 'posts.parquet')
    gold = pd.read_parquet(processed / 'gold.parquet').set_index('row_id')
    tr = posts[posts['split'] == 'train'].copy()
    tr['risk'] = tr['row_id'].map(gold['risk_train'])
    return tr

def match_reals(synth: list[dict], tr: pd.DataFrame) -> tuple[list[dict], list[float]]:
    edges = list(np.quantile(tr['n_ws_tokens'], np.linspace(0.1, 0.9, 9)))
    tr = tr.copy()
    tr['dec'] = tr['n_ws_tokens'].map(lambda n: _length_decile(n, edges))
    by_cell: dict = {}
    for t in tr.itertuples():
        by_cell.setdefault((t.risk, t.dec), []).append((t.post_raw, t.anon_user_id, t.row_id))
    for k in by_cell:
        by_cell[k].sort(key=lambda x: (x[0], x[1]))
    decs_by_risk = {r: sorted((d for rr, d in by_cell if rr == r)) for r in {k[0] for k in by_cell}}
    used: dict = {k: 0 for k in by_cell}
    reals = []
    for i, s in enumerate(synth):
        decs = decs_by_risk.get(s['risk'])
        if not decs:
            continue
        dec = _length_decile(s['n_tokens'], edges)
        key = (s['risk'], min(decs, key=lambda d: (abs(d - dec), d)))
        cell = by_cell[key]
        post, user, rid = cell[used[key] % len(cell)]
        used[key] += 1
        reals.append({'text': post, 'user': user, 'row_id': rid, 'risk': s['risk'], 'dec': key[1], 'synth_idx': i})
    return (reals, edges)

def build_matched(pool_records: dict, prov_by_id: dict, processed: Path) -> dict | None:
    synth = _collect_synth(pool_records, prov_by_id)
    if not synth:
        return None
    reals, edges = match_reals(synth, train_pool(processed))
    texts = [s['text'] for s in synth] + [r['text'] for r in reals]
    y = np.array([1] * len(synth) + [0] * len(reals))
    groups = [s['user'] for s in synth] + [r['user'] for r in reals]
    return {'synth': synth, 'reals': reals, 'texts': texts, 'y': y, 'groups': groups, 'edges': edges}

def default_vectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion
    return FeatureUnion([('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2)), ('word', TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2))])

def grouped_cv_auc(X, y, groups, n_splits: int=5) -> tuple[list[float], np.ndarray]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=n_splits)
    aucs = []
    oof = np.full(len(y), np.nan)
    for tr_i, te_i in gkf.split(X, y, groups):
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(X[tr_i], y[tr_i])
        p = clf.predict_proba(X[te_i])[:, 1]
        oof[te_i] = p
        if len(set(y[te_i])) == 2:
            aucs.append(float(roc_auc_score(y[te_i], p)))
    return (aucs, oof)

def check(pool_records: dict, prov_by_id: dict, processed: Path, out_path: Path | None, seed: int=SC.SYNTH_SEED) -> dict:
    m = build_matched(pool_records, prov_by_id, processed)
    if m is None:
        return {'n_synth': 0, 'pass': True, 'note': 'no synthetic rows'}
    synth, reals = (m['synth'], m['reals'])
    texts, y, groups = (m['texts'], m['y'], m['groups'])
    from sklearn.linear_model import LogisticRegression
    if len(set(groups)) < 5 or min(int((y == 0).sum()), int((y == 1).sum())) < 5:
        result = {'n_synth': len(synth), 'n_real': len(reals), 'pass': True, 'note': 'too few samples or groups for grouped AUC'}
        if out_path is not None:
            state.atomic_write_json(out_path, result)
        return result
    vec = default_vectorizer()
    X = vec.fit_transform(texts)
    aucs, _ = grouped_cv_auc(X, y, groups)
    auc_mean = float(np.mean(aucs)) if aucs else 0.5
    full = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
    names = np.array(vec.get_feature_names_out())
    coef = full.coef_[0]
    top_synth = names[np.argsort(coef)[-20:][::-1]].tolist()
    top_real = names[np.argsort(coef)[:20]].tolist()
    result = {'n_synth': len(synth), 'n_real': len(reals), 'auc_per_fold': [round(a, 4) for a in aucs], 'auc_mean': round(auc_mean, 4), 'threshold': SC.REALISM_MAX_AUC, 'pass': auc_mean <= SC.REALISM_MAX_AUC, 'top_synthetic_features': top_synth, 'top_real_features': top_real, 'matched_sampling_seed': f'{seed}|realism'}
    if out_path is not None:
        state.atomic_write_json(out_path, result)
    return result
