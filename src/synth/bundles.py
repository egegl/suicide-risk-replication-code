"""Real training rows with their labels, folds, and evidence slices (cases for the generated data are drawn from these)."""
from pathlib import Path
import pandas as pd
from prep import constants as PC
from prep import textnorm

def load_rows(processed: Path) -> dict:
    gold = pd.read_parquet(processed / 'gold.parquet')
    posts = pd.read_parquet(processed / 'posts.parquet')
    spans = pd.read_parquet(processed / 'spans.parquet')
    sup: dict[str, list] = {}
    sup_rows = spans[spans['use_for_supervision']].sort_values(['row_id', 'scorer_start', 'scorer_end'])
    for s in sup_rows.itertuples():
        toklen = len(textnorm.scorer_norm(s.scorer_slice).split())
        sup.setdefault(s.row_id, []).append((s.scorer_slice, toklen))
    tr_posts = posts[posts['split'] == 'train'].set_index('row_id')
    rows = []
    for g in gold[gold['is_representative']].itertuples():
        p = tr_posts.loc[g.row_id]
        slices = sup.get(g.row_id, [])
        rows.append({'row_id': g.row_id, 'anon_user_id': p['anon_user_id'], 'fold': int(g.fold), 'risk': g.risk_train, 'factors': tuple(sorted(set(g.factors_set), key=PC.FACTOR_INDEX.__getitem__)), 'n_tokens': int(p['n_ws_tokens']), 'span_lens': tuple((t for _, t in slices)), 'evidence_slices': tuple((s for s, _ in slices)), 'post_raw': p['post_raw']})
    rows.sort(key=lambda r: r['row_id'])
    user_counts = sorted(posts[posts['split'] == 'train'].groupby('anon_user_id').size().sort_index().tolist())
    return {'rows': rows, 'user_counts': user_counts}