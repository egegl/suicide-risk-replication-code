#!/usr/bin/env python3
'Build grouped folds and model inputs.'
import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from prep import constants as C
from prep import align, clean, dedup, emit, folds, hard_slice, load, textnorm, validate
import scorer
_TOKEN_RE = re.compile('\\S+')

def _add_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={'post': 'post_raw'}).copy()
    norms, starts_col, ends_col = ([], [], [])
    for p in df['post_raw']:
        n, s, e = textnorm.normalize_with_map(p)
        norms.append(n)
        starts_col.append(s)
        ends_col.append(e)
    df['post_norm'] = norms
    df['n2r_start'] = starts_col
    df['n2r_end'] = ends_col
    df['n_chars'] = df['post_raw'].str.len().astype('int32')
    df['n_ws_tokens'] = df['post_raw'].map(lambda p: len(_TOKEN_RE.findall(p))).astype('int32')
    df['user_n_posts'] = df.groupby('anon_user_id')['row_id'].transform('count').astype('int32')
    return df

def _scorer_fields(a: align.AlignedSpan, nc: align.NormCache, suppress: bool=False) -> dict:
    if not suppress:
        if a.matched and align.official_match(a.raw_slice, a.text_gold):
            return dict(scorer_start=a.raw_start, scorer_end=a.raw_end, scorer_slice=a.raw_slice, scorer_rung='direct', scorer_matchable=True)
        hit = align.scorer_matchable_slice(a.text_gold, nc)
        if hit:
            s, e = hit
            return dict(scorer_start=s, scorer_end=e, scorer_slice=nc.raw[s:e], scorer_rung='lcs', scorer_matchable=True)
    return dict(scorer_start=-1, scorer_end=-1, scorer_slice='', scorer_rung='', scorer_matchable=False)

def _align_all(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in train.itertuples():
        if not r.gold_eval_spans:
            continue
        nc = align.NormCache(r.post_raw)
        next_idx = len(r.gold_eval_spans)
        text_seen: dict[str, int] = {}
        for i, span in enumerate(r.gold_eval_spans):
            results = align.align_span(span, nc)
            parent, children = (results[0], results[1:])
            k = text_seen.get(span, 0)
            text_seen[span] = k + 1
            if parent.matched and parent.rung in ('exact', 'casefold') and (k > 0) and (len(parent.all_starts) > 1):
                occ = parent.all_starts[min(k, len(parent.all_starts) - 1)]
                length = parent.raw_end - parent.raw_start
                parent.raw_start, parent.raw_end = (occ, occ + length)
                parent.raw_slice = nc.raw[occ:occ + length]
            matchable_children = []
            child_rows = []
            for ch in children:
                cf = _scorer_fields(ch, nc)
                if cf['scorer_matchable']:
                    matchable_children.append(ch)
                child_rows.append((ch, cf))
            parent_fields = _scorer_fields(parent, nc, suppress=parent.rung == 'split' and bool(matchable_children))
            rows.append(dict(row_id=r.row_id, span_idx=i, parent_span_idx=-1, text_gold=parent.text_gold, matched=parent.matched, rung=parent.rung, raw_start=parent.raw_start, raw_end=parent.raw_end, raw_slice=parent.raw_slice, n_occurrences=parent.n_occurrences, all_starts=list(parent.all_starts), use_for_supervision=parent_fields['scorer_matchable'], **parent_fields))
            for ch, cf in child_rows:
                rows.append(dict(row_id=r.row_id, span_idx=next_idx, parent_span_idx=i, text_gold=ch.text_gold, matched=ch.matched, rung=ch.rung, raw_start=ch.raw_start, raw_end=ch.raw_end, raw_slice=ch.raw_slice, n_occurrences=ch.n_occurrences, all_starts=list(ch.all_starts), use_for_supervision=cf['scorer_matchable'], **cf))
                next_idx += 1
    return pd.DataFrame(rows)

def _scorer_smoke(train: pd.DataFrame, spans: pd.DataFrame) -> dict:
    ids = train['row_id'].tolist()
    golds = train['gold_eval_spans'].map(list).tolist()
    assert scorer.phrase_f1(golds, golds) == 1.0
    empty = scorer.phrase_f1([[] for _ in ids], golds)
    assert abs(empty - C.EXPECTED['empty_pred_phrase_f1']) < 1e-09, empty
    sup: dict[str, list[str]] = {}
    for s in spans[spans['use_for_supervision']].sort_values(['row_id', 'scorer_start']).itertuples():
        sup.setdefault(s.row_id, []).append(s.scorer_slice)
    preds = [sup.get(rid, []) for rid in ids]
    ceiling = scorer.phrase_f1(preds, golds)
    assert ceiling >= C.EXPECTED['min_scorer_ceiling'], ceiling
    assert scorer.phrase_f1_post(['just wanna die', 'just wanna fucking die'], ['just wanna die', 'just wanna fucking die']) == 1.0
    assert scorer.match('kill myself', 'I want to kill myself right now')
    assert not scorer.match('a b c d', 'die')
    return {'empty_pred_phrase_f1': empty, 'scorer_slice_ceiling_phrase_f1': ceiling}

def build(out_dir: Path, keep_indicator_evidence: bool=False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_raw, lb_raw = load.load_raw()
    train = train_raw.copy()
    train['risk_norm'] = clean.normalize_risk(train['suicide risk'])
    train['risk_ord'] = train['risk_norm'].map(C.RISK_ORD).astype('int8')
    train = clean.evidence_columns(train)
    train = clean.factors_columns(train)
    train = _add_text_columns(train)
    lb = _add_text_columns(lb_raw)
    spans = _align_all(train)
    train = dedup.add_dup_columns(train.rename(columns={'post_raw': 'post'})).rename(columns={'post': 'post_raw'})
    train = dedup.resolve_conflicts(train)
    lb = dedup.add_dup_columns(lb.rename(columns={'post_raw': 'post'})).rename(columns={'post': 'post_raw'})
    lb['dup_group_id'] += int(train['dup_group_id'].max()) + 1
    lb['near_dup_cluster_id'] += int(train['near_dup_cluster_id'].max()) + 1
    lb['is_representative'] = ~lb.duplicated(subset='dup_group_id')
    lb['is_label_conflict'] = False
    lb['cv_component'] = -1
    lb['fold'] = -1
    fold_df = folds.assign_folds(train)
    train = train.merge(fold_df[['row_id', 'fold']], on='row_id', validate='1:1')
    train['fold'] = train['fold'].astype('int8')
    train['split'] = 'train'
    lb['split'] = 'leaderboard'
    emit.write_posts(out_dir, train, lb)
    emit.write_gold(out_dir, train)
    emit.write_spans(out_dir, spans)
    emit.write_folds(out_dir, fold_df)
    emit.write_constraints(out_dir, lb)
    sft_stats = emit.write_sft(out_dir, train, lb, spans, keep_indicator_evidence=keep_indicator_evidence)
    assert sft_stats['sft_skipped_unalignable'] <= C.EXPECTED['max_sft_skipped_unalignable'], sft_stats
    assert sft_stats['sft_skipped_truncated_out'] <= C.EXPECTED['max_sft_skipped_truncated'], sft_stats
    hs = hard_slice.build(train, out_dir)
    report = validate.run_all(train, lb, spans, out_dir)
    smoke = _scorer_smoke(train, spans)
    parents = spans[spans['parent_span_idx'] == -1]
    meta = {'inputs_sha256': C.RAW_SHA256, 'target_schema': {'primary': C.TARGET_SCHEMA, 'legacy': [s for s in C.TARGET_SCHEMAS if s != C.TARGET_SCHEMA]}, 'counts': {'train_rows': len(train), 'lb_rows': len(lb), 'gold_spans': int(train['gold_eval_spans'].map(len).sum()), 'supervision_spans': int(spans['use_for_supervision'].sum()), 'scorer_rung_census': {k: int(v) for k, v in spans[spans['use_for_supervision']]['scorer_rung'].value_counts().items()}, 'ladder_census': {k: int(v) for k, v in parents[parents['matched']]['rung'].value_counts().items()}, 'unmatched_spans': int((~parents['matched']).sum()), 'gap_fragments': int((spans['parent_span_idx'] >= 0).sum()), 'multi_occurrence_spans': int((parents['matched'] & (parents['n_occurrences'] > 1)).sum()), 'near_dup_clusters_multi': int((train.groupby('near_dup_cluster_id').size() > 1).sum()), 'risk_train_overrides': int((train['risk_train'] != train['risk_norm']).sum()), 'hard_slice_curated': len(hs['curated']), 'hard_slice_mined': {k: len(v) for k, v in hs['mined_review_candidates'].items()}, **sft_stats}, 'scorer_smoke': smoke, 'fold_sizes': {int(k): int(v) for k, v in train.groupby('fold').size().items()}, 'warnings': report.warnings, 'checks': {k: v['pass'] for k, v in report.checks.items()}}
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2, default=str))
    (out_dir / 'validation_report.json').write_text(json.dumps({'checks': report.checks, 'warnings': report.warnings}, indent=2, default=str))
    return meta

def _dir_hashes(d: Path) -> dict[str, str]:
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob('*')) if p.is_file()}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/processed')
    ap.add_argument('--check-determinism', action='store_true')
    ap.add_argument('--refreeze-folds', action='store_true')
    ap.add_argument('--keep-indicator-evidence', action='store_true')
    args = ap.parse_args()
    final = Path(args.out)
    staging = final.parent / (final.name + '.staging')
    if staging.exists():
        shutil.rmtree(staging)
    meta = build(staging, keep_indicator_evidence=args.keep_indicator_evidence)
    if args.check_determinism:
        staging2 = final.parent / (final.name + '.staging2')
        if staging2.exists():
            shutil.rmtree(staging2)
        build(staging2, keep_indicator_evidence=args.keep_indicator_evidence)
        h1, h2 = (_dir_hashes(staging), _dir_hashes(staging2))
        diff = {k for k in h1.keys() | h2.keys() if h1.get(k) != h2.get(k)}
        assert not diff, f'non-deterministic artifacts: {sorted(diff)}'
        shutil.rmtree(staging2)
        print('determinism check passed: all artifact hashes identical across two builds')
    frozen = final / 'folds.parquet'
    if frozen.exists() and (not args.refreeze_folds):
        old = pd.read_parquet(frozen).set_index('row_id')['fold']
        new = pd.read_parquet(staging / 'folds.parquet').set_index('row_id')['fold']
        if not old.equals(new):
            raise SystemExit('fold assignments changed vs frozen folds.parquet — rerun with --refreeze-folds if intentional')
    if final.exists():
        shutil.rmtree(final)
    staging.rename(final)
    print(f'build OK -> {final}')
    print(json.dumps(meta['counts'], indent=2, default=str))
    print('scorer smoke:', meta['scorer_smoke'])
    if meta['warnings']:
        print('warnings:')
        for w in meta['warnings']:
            print(' -', w)
if __name__ == '__main__':
    main()
