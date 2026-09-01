from pathlib import Path
import pandas as pd
import scorer
from prep import align
from synth import state
from . import constants as C

def _train_df(processed: Path) -> pd.DataFrame:
    cols = ['row_id', 'split', 'near_dup_cluster_id', 'is_representative', 'post_raw']
    df = pd.read_parquet(Path(processed) / 'posts.parquet', columns=cols)
    return df[df['split'] == 'train'].reset_index(drop=True)

def rep_map(processed: Path) -> dict[str, str]:
    df = _train_df(processed)
    rep_by_cluster: dict[int, str] = {}
    for r in df[df['is_representative']].itertuples():
        assert r.near_dup_cluster_id not in rep_by_cluster, f'cluster {r.near_dup_cluster_id} has multiple representatives'
        rep_by_cluster[r.near_dup_cluster_id] = r.row_id
    out = {r.row_id: rep_by_cluster[r.near_dup_cluster_id] for r in df.itertuples()}
    assert len(out) == len(df), 'duplicate row_id in posts.parquet train split'
    return out

def _realigned_spans(spans: list[str], raw_post: str) -> list[str]:
    nc = align.NormCache(raw_post)
    out = []
    for sp in spans:
        hit = align.align_generated(sp, nc)
        if hit is not None:
            s, e = hit
            out.append(nc.raw[s:e])
    return out

def _fallback_pred() -> dict:
    return {'risk': C.FALLBACK_PRED['risk'], 'spans': list(C.FALLBACK_PRED['spans']), 'factors': list(C.FALLBACK_PRED['factors'])}

def _propagate(preds: dict[str, dict], need_ids, processed: Path) -> dict[str, dict]:
    df = _train_df(processed)
    rmap = rep_map(processed)
    raw_by_row = dict(zip(df['row_id'], df['post_raw']))
    out: dict[str, dict] = {}
    missing: list[str] = []
    for rid in need_ids:
        if rid in preds:
            p = preds[rid]
            out[rid] = {'risk': p['risk'], 'spans': list(p['spans']), 'factors': list(p['factors'])}
            continue
        rep = rmap.get(rid)
        if rep in preds:
            p = preds[rep]
            out[rid] = {'risk': p['risk'], 'spans': _realigned_spans(list(p['spans']), raw_by_row[rid]), 'factors': list(p['factors'])}
        elif rid in C.ORPHAN_GOLD_ROWS:
            out[rid] = _fallback_pred()
        else:
            missing.append(rid)
    if missing:
        raise ValueError(f'propagation left {len(missing)} gold rows uncovered (their cluster representative has no prediction), e.g. {sorted(missing)[:5]}')
    return out

def propagate(preds: dict[str, dict], processed: Path) -> dict[str, dict]:
    df = _train_df(processed)
    return _propagate(preds, list(df['row_id']), processed)

def _score_both(preds: dict[str, dict], gold: dict[str, dict]) -> dict:
    covered = {i: preds[i] for i in gold}
    ee1 = scorer.score(covered, gold, empty_convention='ee1')
    skip = scorer.score(covered, gold, empty_convention='skip')
    return {'composite_ee1': ee1['composite'], 'composite_skip': skip['composite'], 'risk_wf1': ee1['risk_wf1'], 'phrase_f1_ee1': ee1['phrase_f1'], 'phrase_f1_skip': skip['phrase_f1'], 'macro_f1': ee1['macro_f1'], 'n_posts': ee1['n_posts']}

def score_fold(preds: dict[str, dict], fold: int, repo_root: Path) -> dict:
    processed = Path(repo_root) / 'data' / 'processed'
    gold = scorer.load_gold(processed / 'gold.jsonl', fold=fold)
    expanded = _propagate(preds, sorted(gold), processed)
    return _score_both(expanded, gold)

def pool_runs(runs_dir: Path, arm: str, seed: int, repo_root: Path) -> dict:
    runs_dir = Path(runs_dir)
    repo_root = Path(repo_root)
    processed = repo_root / 'data' / 'processed'
    rows: list[dict] = []
    for k in range(C.N_FOLDS):
        path = runs_dir / f'{arm}-f{k}-s{seed}' / 'oof.jsonl'
        fold_rows = state.read_jsonl(path)
        for r in fold_rows:
            assert r['fold'] == k, (path, r['row_id'], r['fold'])
        rows.extend(fold_rows)
    sft_ids = {r['row_id'] for r in state.read_jsonl(repo_root / C.SFT_DIR / 'full_train.jsonl')}
    ids = [r['row_id'] for r in rows]
    dupes = sorted({i for i in ids if ids.count(i) > 1}) if len(ids) != len(set(ids)) else []
    assert not dupes, f"rows appear in multiple folds' oof files, e.g. {dupes[:5]}"
    assert set(ids) == sft_ids, f'pooled OOF covers {len(set(ids))} rows, expected the {len(sft_ids)} SFT val rows (missing e.g. {sorted(sft_ids - set(ids))[:5]}, extra e.g. {sorted(set(ids) - sft_ids)[:5]})'
    rows.sort(key=lambda r: (r['fold'], r['row_id']))
    out_path = runs_dir / 'pooled' / f'{arm}-s{seed}.oof.jsonl'
    state.atomic_write_jsonl(out_path, rows)
    preds = {r['row_id']: {'risk': r['risk'], 'spans': r['spans'], 'factors': r['factors']} for r in rows}
    full = propagate(preds, processed)
    gold = scorer.load_gold(processed / 'gold.jsonl')
    return _score_both(full, gold)
