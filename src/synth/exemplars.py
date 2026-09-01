import json
from prep import constants as PC
from . import constants as SC

def build_pools(rows: list[dict]) -> dict:
    pools: dict = {}
    for pool in SC.POOLS:
        pools[pool] = {}
        for factor in SC.RARE_FACTORS:
            slug = SC.FACTOR_SLUGS[factor]
            elig = [r for r in rows if factor in r['factors'] and (pool == 'full' or r['fold'] != int(pool))]
            pools[pool][slug] = [{'row_id': r['row_id'], 'fold': r['fold'], 'risk': r['risk'], 'factors': list(r['factors']), 'evidence': list(r['evidence_slices']), 'post': r['post_raw']} for r in sorted(elig, key=lambda r: r['row_id'])]
    return pools

def draw(pool_exemplars: list[dict], rng, n: int=3) -> list[str]:
    with_spans = [e['row_id'] for e in pool_exemplars if e['evidence']]
    without = [e['row_id'] for e in pool_exemplars if not e['evidence']]
    rng.shuffle(with_spans)
    rng.shuffle(without)
    return (with_spans + without)[:n]

def render(exemplar: dict) -> str:
    lines = [f"POST: {exemplar['post']}", f"RISK: {exemplar['risk']}", f"FACTORS: {json.dumps(exemplar['factors'], ensure_ascii=False)}", f"EVIDENCE: {json.dumps(exemplar['evidence'], ensure_ascii=False)}"]
    return '\n'.join(lines)

def validate_pools(pools: dict) -> None:
    for pool, by_slug in pools.items():
        for slug, exs in by_slug.items():
            factor = SC.SLUG_FACTORS[slug]
            for e in exs:
                assert factor in e['factors'], (pool, slug, e['row_id'])
                if pool != 'full':
                    assert e['fold'] != int(pool), (pool, slug, e['row_id'])
                assert e['risk'] in PC.RISK_CLASSES
