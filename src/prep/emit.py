import json
from pathlib import Path
import pandas as pd
from . import constants as C
from . import context
from . import target as target_schema

def _jsonl(path: Path, records: list[dict]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def write_posts(out: Path, train: pd.DataFrame, lb: pd.DataFrame) -> pd.DataFrame:
    cols = ['row_id', 'split', 'anon_user_id', 'post_id', 'post_raw', 'post_norm', 'n2r_start', 'n2r_end', 'n_chars', 'n_ws_tokens', 'user_n_posts', 'dup_group_id', 'near_dup_cluster_id', 'cv_component', 'is_representative', 'is_label_conflict', 'fold']
    posts = pd.concat([train[cols], lb[cols]], ignore_index=True)
    posts.to_parquet(out / 'posts.parquet', index=False)
    return posts

def write_gold(out: Path, train: pd.DataFrame) -> None:
    cols = ['row_id', 'risk_norm', 'risk_ord', 'risk_train', 'gold_eval_spans', 'has_evidence', 'factors_mentions', 'factors_set', *[f'f_{j:02d}' for j in range(len(C.FACTORS_24))], 'train_weight', 'is_representative', 'is_label_conflict', 'fold']
    train[cols].to_parquet(out / 'gold.parquet', index=False)
    records = [{'row_id': r.row_id, 'fold': int(r.fold), 'risk': r.risk_norm, 'spans': list(r.gold_eval_spans), 'factors': list(r.factors_set)} for r in train.itertuples()]
    _jsonl(out / 'gold.jsonl', records)

def write_spans(out: Path, spans: pd.DataFrame) -> None:
    spans.to_parquet(out / 'spans.parquet', index=False)

def write_folds(out: Path, folds: pd.DataFrame) -> None:
    folds.to_parquet(out / 'folds.parquet', index=False)

def write_constraints(out: Path, lb: pd.DataFrame) -> None:
    groups = [sorted(g['row_id']) for _, g in lb.groupby('post_raw', sort=False) if len(g) > 1]
    assert sorted(map(tuple, groups)) == sorted((tuple(sorted(p)) for p in C.LB_DUP_PAIRS)), groups
    payload = {'identical_groups': sorted(map(list, groups)), 'known_labels': C.LB_KNOWN, 'consistency_blocks': {'U0123': list(C.U0123_BLOCK), 'U0161': list(C.U0161_BLOCK)}}
    (out / 'constraints.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False))

def _evidence_target(risk_train: str, sup: list[dict], kept_ranges: list, keep_indicator_evidence: bool) -> list[str] | None:
    if risk_train == 'Indicator' and (not keep_indicator_evidence):
        return []
    sup = [s for s in sup if any((s['start'] >= a and s['end'] <= b for a, b in kept_ranges))]
    sup.sort(key=lambda s: (s['start'], s['end']))
    out = [s['slice'] for s in sup]
    if risk_train == 'Indicator':
        return out
    return out or None

def write_sft(out: Path, train: pd.DataFrame, lb: pd.DataFrame, spans: pd.DataFrame, keep_indicator_evidence: bool=False) -> dict:
    sup_by_row: dict[str, list[dict]] = {}
    for s in spans[spans['use_for_supervision']].itertuples():
        sup_by_row.setdefault(s.row_id, []).append({'start': int(s.scorer_start), 'end': int(s.scorer_end), 'slice': s.scorer_slice})

    def build_rows(df: pd.DataFrame, with_target: bool, schema: str, skip_stats: dict) -> list[dict]:
        records = []
        for _, user_df in df.groupby('anon_user_id', sort=False):
            user_df = user_df.sort_values('post_id')
            texts = user_df['post_raw'].tolist()
            for pos, row in enumerate(user_df.itertuples()):
                if with_target and (not row.is_representative):
                    continue
                sup = sup_by_row.get(row.row_id, []) if with_target else []
                cover = [(s['start'], s['end']) for s in sup] or None
                input_text, kept_ranges, is_trunc = context.build_input(texts, pos, cover=cover, instruction=target_schema.instruction_for(schema))
                rec = {'row_id': row.row_id, 'input': input_text, 'is_truncated': is_trunc}
                if with_target:
                    ev = _evidence_target(row.risk_train, sup, kept_ranges, keep_indicator_evidence)
                    if ev is None:
                        skip_stats['truncated_out' if sup else 'unalignable'] += 1
                        continue
                    rec['fold'] = int(row.fold)
                    rec['weight'] = float(row.train_weight)
                    rec['target'] = target_schema.render_target(row.risk_train, ev, list(row.factors_set), schema)
                records.append(rec)
        return records
    stats_by_schema = {}
    for schema in C.TARGET_SCHEMAS:
        sft_dir = out / ('sft' if schema == C.TARGET_SCHEMA else f'sft_{schema}')
        sft_dir.mkdir(exist_ok=True)
        skip_stats = {'unalignable': 0, 'truncated_out': 0}
        all_records = build_rows(train, with_target=True, schema=schema, skip_stats=skip_stats)
        for k in range(C.N_FOLDS):
            tr = [r for r in all_records if r['fold'] != k]
            va = [r for r in all_records if r['fold'] == k]
            _jsonl(sft_dir / f'train_fold{k}.jsonl', tr)
            _jsonl(sft_dir / f'val_fold{k}.jsonl', va)
        _jsonl(sft_dir / 'full_train.jsonl', all_records)
        lb_records = build_rows(lb, with_target=False, schema=schema, skip_stats=skip_stats)
        _jsonl(sft_dir / 'leaderboard_infer.jsonl', lb_records)
        stats_by_schema[schema] = {'sft_rows': len(all_records), 'sft_skipped_unalignable': skip_stats['unalignable'], 'sft_skipped_truncated_out': skip_stats['truncated_out'], 'lb_infer_rows': len(lb_records)}
    others = [s for s in C.TARGET_SCHEMAS if s != C.TARGET_SCHEMA]
    for s in others:
        assert stats_by_schema[s] == stats_by_schema[C.TARGET_SCHEMA], stats_by_schema
    return stats_by_schema[C.TARGET_SCHEMA]
