import random
from pathlib import Path
import pandas as pd
from prep import constants as PC
from prep import textnorm
from . import constants as SC
from . import exemplars, state

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
        rows.append({'row_id': g.row_id, 'fold': int(g.fold), 'risk': g.risk_train, 'factors': tuple(sorted(set(g.factors_set), key=PC.FACTOR_INDEX.__getitem__)), 'n_tokens': int(p['n_ws_tokens']), 'span_lens': tuple((t for _, t in slices)), 'evidence_slices': tuple((s for s, _ in slices)), 'post_raw': p['post_raw']})
    rows.sort(key=lambda r: r['row_id'])
    user_counts = sorted(posts[posts['split'] == 'train'].groupby('anon_user_id').size().sort_index().tolist())
    return {'rows': rows, 'user_counts': user_counts}

def _weighted_choice(rng: random.Random, pairs) -> int:
    x, acc = (rng.random(), 0.0)
    for value, p in pairs:
        acc += p
        if x < acc:
            return value
    return pairs[-1][0]

def _style_persona(rng: random.Random) -> dict:
    return {k: rng.random() < p for k, p in SC.STYLE_PERSONA.items()}

def _comma_p(len_target: int) -> float:
    for edge, p in SC.STYLE_COMMA_BY_LEN:
        if edge is None or len_target < edge:
            return p
    return SC.STYLE_COMMA_BY_LEN[-1][1]

def _style_post(rng: random.Random, persona: dict, len_target: int) -> dict:
    s = {k: rng.random() < p for k, p in SC.STYLE_POST.items()}
    anchor = _weighted_choice(rng, SC.STYLE_ANCHORS)
    s['anchor'] = anchor if s['future_ref'] else ''
    s['comma'] = rng.random() < _comma_p(len_target)
    if persona['no_terminal_punct']:
        s['exclamation'] = False
        s['ellipsis'] = False
    return s

def _span_profile(rng: random.Random, risk: str, rows_by_risk_with_spans: dict) -> tuple[list[int], str | None]:
    if risk == 'Indicator':
        return ([], None)
    b = rng.choice(rows_by_risk_with_spans[risk])
    lens = list(b['span_lens'])[:SC.SPAN_COUNT_MAX]
    return ([min(t, SC.SPAN_TOKENS_MAX) for t in lens], b['row_id'])

def _post_spec(rng: random.Random, src: dict, risk: str, rows_by_risk_with_spans: dict) -> dict:
    span_targets, src_span_row = ([], None)
    if risk != 'Indicator':
        if src['risk'] == risk and src['span_lens']:
            span_targets, src_span_row = ([min(t, SC.SPAN_TOKENS_MAX) for t in src['span_lens'][:SC.SPAN_COUNT_MAX]], src['row_id'])
        else:
            span_targets, src_span_row = _span_profile(rng, risk, rows_by_risk_with_spans)
    return {'risk': risk, 'factors': list(src['factors']), 'n_spans': len(span_targets), 'span_token_targets': span_targets, 'len_target_tokens': max(SC.POST_LEN_CLAMP[0], min(src['n_tokens'], SC.POST_LEN_CLAMP[1])), 'src_label_row': src['row_id'], 'src_span_row': src_span_row}

def build(processed: Path) -> tuple[list[dict], dict]:
    data = load_rows(processed)
    rows = data['rows']
    pools = exemplars.build_pools(rows)
    exemplars.validate_pools(pools)
    bundles: list[dict] = []
    su_serial = 0
    quota = SC.QUOTA_FULL
    for pool in SC.POOLS:
        elig = [r for r in rows if pool == 'full' or r['fold'] != int(pool)]
        by_risk_spans = {risk: [r for r in elig if r['risk'] == risk and r['span_lens']] for risk in PC.RISK_CLASSES if risk != 'Indicator'}
        for factor in SC.RARE_FACTORS:
            slug = SC.FACTOR_SLUGS[factor]
            carriers = [r for r in elig if factor in r['factors']]
            low_sev = [r for r in carriers if r['risk'] in ('Indicator', 'Ideation')]
            assert carriers, (pool, factor)
            ctx_pool = elig
            serial, targets_done, ba_count = (0, 0, 0)
            while targets_done < quota:
                su_serial += 1
                assert su_serial <= 9999, 'SU id space exhausted'
                bundle_id = f"{SC.PROMPT_VERSION}-k{('f' if pool == 'full' else pool)}-{slug}-{serial:04d}"
                seed_key = f'{SC.SYNTH_SEED}|{SC.PROMPT_VERSION}|pool{pool}|{slug}|{serial}'
                rng = random.Random(seed_key)
                persona_style = _style_persona(rng)
                L = rng.randint(*SC.TIMELINE_LEN)
                declared_n = max(rng.choice(data['user_counts']), L)
                n_target = min(_weighted_choice(rng, SC.N_TARGET_PROBS), L - 1)
                target_pos = sorted(rng.sample(range(L - 1), n_target))
                posts = []
                for p in range(L):
                    if p in target_pos:
                        a = rng.choice(carriers)
                        risk = a['risk']
                        if risk in ('Behavior', 'Attempt') and ba_count + 1 > SC.MAX_BA_SHARE * (targets_done + 1):
                            a = rng.choice(low_sev) if low_sev else a
                            risk = a['risk'] if low_sev else 'Ideation'
                        if risk in ('Behavior', 'Attempt'):
                            ba_count += 1
                        spec = _post_spec(rng, a, risk, by_risk_spans)
                        spec['is_target'] = True
                        targets_done += 1
                    else:
                        a = rng.choice(ctx_pool)
                        for _ in range(50):
                            if factor not in a['factors']:
                                break
                            a = rng.choice(ctx_pool)
                        assert factor not in a['factors'], (bundle_id, p)
                        spec = _post_spec(rng, a, a['risk'], by_risk_spans)
                        spec['is_target'] = False
                    spec['pos'] = p
                    spec['emit_record'] = p <= L - 2
                    spec['style'] = _style_post(rng, persona_style, spec['len_target_tokens'])
                    posts.append(spec)
                bundles.append({'bundle_id': bundle_id, 'pool': pool, 'target_factor': factor, 'su_id': f'SU{su_serial:04d}', 'declared_n': declared_n, 'n_posts': L, 'persona_style': persona_style, 'posts': posts, 'exemplar_row_ids': exemplars.draw(pools[pool][slug], rng), 'seed_key': seed_key})
                serial += 1
    return (bundles, pools)

def write(bundles: list[dict], pools: dict, out_bundles: Path, out_exemplars: Path) -> None:
    state.atomic_write_jsonl(out_bundles, bundles)
    state.atomic_write_json(out_exemplars, pools)
