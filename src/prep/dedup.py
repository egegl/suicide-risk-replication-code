import re
import pandas as pd
from . import constants as C
from . import textnorm

class UnionFind:

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = (self.find(a), self.find(b))
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

def near_dup_key(raw: str) -> str:
    key = textnorm.normalize(raw)
    key = re.sub('\\.{2,}', '.', key)
    key = re.sub('[.!?\\s"]+$', '', key)
    return key

def _group_ids(uf: UnionFind, index: pd.Index) -> pd.Series:
    roots = [uf.find(i) for i in range(len(index))]
    ids = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return pd.Series([ids[r] for r in roots], index=index)

def add_dup_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    pos = {rid: i for i, rid in enumerate(df['row_id'])}
    df['dup_group_id'] = pd.factorize(df['post'])[0]
    uf = UnionFind(len(df))
    for _, g in df.groupby(df['post'].map(near_dup_key), sort=False):
        idx = g.index.tolist()
        for j in idx[1:]:
            uf.union(idx[0], j)
    for cluster in C.NEAR_DUP_MERGES:
        ids = [pos[r] for r in cluster if r in pos]
        for j in ids[1:]:
            uf.union(ids[0], j)
    df['near_dup_cluster_id'] = _group_ids(uf, df.index)
    uf2 = UnionFind(len(df))
    for _, g in df.groupby('anon_user_id', sort=False):
        idx = g.index.tolist()
        for j in idx[1:]:
            uf2.union(idx[0], j)
    for _, g in df.groupby('near_dup_cluster_id', sort=False):
        idx = g.index.tolist()
        for j in idx[1:]:
            uf2.union(idx[0], j)
    df['cv_component'] = _group_ids(uf2, df.index)
    return df

def resolve_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['risk_train'] = df['risk_norm']
    df['is_representative'] = True
    df['is_label_conflict'] = False
    for _, g in df.groupby('near_dup_cluster_id', sort=False):
        if len(g) == 1:
            continue
        counts = g['risk_norm'].value_counts()
        top = counts.max()
        resolved = max((c for c in counts.index if counts[c] == top), key=lambda c: C.RISK_ORD[c])
        majority_rows = g[g['risk_norm'] == resolved]
        rep = majority_rows['row_id'].min()
        conflict = g['risk_norm'].nunique() > 1
        df.loc[g.index, 'risk_train'] = resolved
        df.loc[g.index, 'is_representative'] = df.loc[g.index, 'row_id'] == rep
        df.loc[g.index, 'is_label_conflict'] = conflict
    exact_sizes = df.groupby('dup_group_id')['row_id'].transform('count')
    df['train_weight'] = (1.0 / exact_sizes).astype('float32')
    for rid, expected in C.PINNED_RESOLUTIONS.items():
        actual = df.loc[df['row_id'] == rid, 'risk_train'].iloc[0]
        assert actual == expected, f'{rid}: resolved {actual}, pinned {expected}'
    return df
