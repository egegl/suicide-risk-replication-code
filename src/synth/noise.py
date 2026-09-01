import random
import re
from prep import align, textnorm
from . import constants as SC
_RE_CONTRACTED = re.compile("(?i)(?:n[’']t|[’'](?:m|re|ve|ll|d|s))(?![A-Za-z])")
_RE_DROPPED = re.compile("(?i)(?<![A-Za-z’'])(?:" + '|'.join(sorted(SC.CONTRACTIONS_DROPPED, key=len, reverse=True)) + ")(?![A-Za-z’'])")
_RE_SPELLED = re.compile("(?i)(?<![A-Za-z’'])(?:" + '|'.join((re.escape(k) for k in sorted(SC.CONTRACTIONS_SPELLED, key=len, reverse=True))) + ")(?![A-Za-z’'])")
_RE_CAP_I = re.compile('(?<![A-Za-z])I(?![A-Za-z])')

def style_flags(text: str) -> dict:
    return {'contracted': bool(_RE_CONTRACTED.search(text)), 'dropped': bool(_RE_DROPPED.search(text)), 'spelled': bool(_RE_SPELLED.search(text)), 'apos': "'" in text or '’' in text, 'curly': '’' in text, 'uppercase': any((c.isupper() for c in text)), 'cap_i': bool(_RE_CAP_I.search(text))}
_RE_PRON_I = re.compile('(?<![A-Za-z])i(?![A-Za-z])')
_RE_SENT_START = re.compile('(?:^|[.!?]\\s+|\\n\\s*)([a-z])')

def _case_sites(text: str) -> list[tuple[int, str, bool]]:
    sites: dict[int, tuple[str, bool]] = {}
    for m in _RE_PRON_I.finditer(text):
        sites[m.start()] = ('I', True)
    for m in _RE_SENT_START.finditer(text):
        pos = m.start(1)
        if pos not in sites:
            sites[pos] = (text[pos].upper(), False)
    return [(pos, ins, pron) for pos, (ins, pron) in sorted(sites.items())]

def _case_pass(text: str, ranges: list[tuple[int, int]], lowercase_i: bool, seed_key: str, log: list) -> tuple[str, list[tuple[int, int]], str | None]:
    if lowercase_i:
        return (text, ranges, None)
    crng = random.Random(seed_key)
    converted = 0
    for pos, ins, pron in reversed(_case_sites(text)):
        if not pron and crng.random() >= SC.NOISE['sentence_cap_given_capitalizer']:
            continue
        if _site_placement(pos, 1, ranges) == 'inside':
            text, ranges = _apply_in_span(text, ranges, pos, 1, ins, log)
        else:
            text, ranges = _apply(text, ranges, pos, 1, ins, log)
        converted += 1
    return (text, ranges, 'case' if converted else None)

def _i_casing(text: str) -> str:
    lower = len(re.findall('(?<![A-Za-z])i(?![A-Za-z])', text))
    upper = len(re.findall('(?<![A-Za-z])I(?![A-Za-z])', text))
    return 'i' if lower > upper else 'I'

def _match_case(matched: str, canon: str, i_case: str) -> str:
    letters = [c for c in matched if c.isalpha()]
    if len(letters) >= 2 and matched.isupper():
        return canon.upper()
    if matched[0].isupper():
        return canon[0].upper() + canon[1:]
    if canon[0] == 'i' and i_case == 'I':
        return 'I' + canon[1:]
    return canon

def _preceding_word(text: str, pos: int) -> str:
    m = re.search('([A-Za-z]+)[^A-Za-z]{0,2}$', text[max(0, pos - 24):pos])
    return m.group(1).lower() if m else ''

def _following_word(text: str, end: int) -> str:
    m = re.match('[ ]+([A-Za-z]+)', text[end:end + 25])
    return m.group(1).lower() if m else ''

def _contract_sites(text: str, mode: str) -> list[tuple[int, int, str]]:
    i_case = _i_casing(text)
    sites: list[tuple[int, int, str]] = []
    if mode == 'contract':
        for m in _RE_DROPPED.finditer(text):
            canon = SC.CONTRACTIONS_DROPPED[m.group(0).lower()]
            sites.append((m.start(), len(m.group(0)), _match_case(m.group(0), canon, i_case)))
    for m in _RE_SPELLED.finditer(text):
        phrase = ' '.join(m.group(0).lower().split())
        canon = SC.CONTRACTIONS_SPELLED[phrase]
        if phrase not in SC.SPELLED_UNGUARDED:
            follower = _following_word(text, m.end())
            if not follower:
                continue
            if phrase == 'i have' and follower not in SC.IVE_FOLLOWERS:
                continue
            if phrase in ('i will', 'i would') and follower == 'not':
                mm = re.match("[ ]+not(?![A-Za-z’'])", text[m.end():])
                if mm:
                    neg = 'will not' if phrase == 'i will' else 'would not'
                    start = m.start() + 2
                    matched = text[start:m.end() + mm.end()]
                    canon2 = SC.CONTRACTIONS_SPELLED[neg]
                    repl2 = canon2.replace("'", '') if mode == 'drop' else canon2
                    sites.append((start, len(matched), _match_case(matched, repl2, i_case)))
                continue
            if phrase not in ('i have', 'i will', 'i would') and _preceding_word(text, m.start()) in SC.SPELLED_BLOCK_PRECEDING:
                continue
        repl = canon.replace("'", '') if mode == 'drop' else canon
        sites.append((m.start(), len(m.group(0)), _match_case(m.group(0), repl, i_case)))
    sites.sort()
    return sites

def _apply_in_span(text: str, ranges: list[tuple[int, int]], pos: int, del_len: int, ins: str, log: list) -> tuple[str, list[tuple[int, int]]]:
    delta = len(ins) - del_len
    moved = []
    for s, e in ranges:
        if s <= pos and pos + del_len <= e:
            moved.append((s, e + delta))
        elif s >= pos + del_len:
            moved.append((s + delta, e + delta))
        else:
            moved.append((s, e))
    log.append({'op': 'edit_in_span', 'pos': pos, 'before': text[pos:pos + del_len], 'after': ins})
    return (text[:pos] + ins + text[pos + del_len:], moved)

def _site_placement(pos: int, del_len: int, ranges: list[tuple[int, int]]) -> str:
    if _outside(pos, del_len, ranges):
        return 'outside'
    if any((s <= pos and pos + del_len <= e for s, e in ranges)):
        return 'inside'
    return 'straddle'

def _contract_pass(text: str, ranges: list[tuple[int, int]], drops_apostrophes: bool, seed_key: str, log: list) -> tuple[str, list[tuple[int, int]], str | None]:
    crng = random.Random(seed_key)
    if drops_apostrophes:
        mode, take = ('drop', lambda: crng.random() < SC.DROP_TOKEN_P)
        op = 'contract_drop'
    else:
        mode = 'contract'
        r = crng.random()
        if r < SC.CONTRACT['full_p']:
            take, op = (lambda: True, 'contract_full')
        elif r < SC.CONTRACT['full_p'] + SC.CONTRACT['partial_p']:
            take, op = (lambda: crng.random() < SC.CONTRACT['token_p'], 'contract_partial')
        else:
            return (text, ranges, None)
    sites = _contract_sites(text, mode)
    converted = 0
    for pos, del_len, ins in reversed(sites):
        if not take():
            continue
        placement = _site_placement(pos, del_len, ranges)
        if placement == 'straddle':
            continue
        if placement == 'inside':
            text, ranges = _apply_in_span(text, ranges, pos, del_len, ins, log)
        else:
            text, ranges = _apply(text, ranges, pos, del_len, ins, log)
        converted += 1
    return (text, ranges, op if converted else None)

def _claim_ranges(text: str, evidence: list[str]) -> list[tuple[int, int]] | None:
    claimed: list[tuple[int, int]] = []
    for ev in evidence:
        start = 0
        while True:
            i = text.find(ev, start)
            if i == -1:
                return None
            j = i + len(ev)
            if not any((i < e and s < j for s, e in claimed)):
                claimed.append((i, j))
                break
            start = i + 1
    return claimed

def _outside(pos: int, length: int, ranges: list[tuple[int, int]]) -> bool:
    a, b = (pos, pos + length)
    return not any((a < e and s < b for s, e in ranges))

def _apply(text: str, ranges: list[tuple[int, int]], pos: int, del_len: int, ins: str, log: list) -> tuple[str, list[tuple[int, int]]]:
    assert _outside(pos, del_len, ranges), (pos, del_len)
    new = text[:pos] + ins + text[pos + del_len:]
    delta = len(ins) - del_len
    moved = [(s + delta, e + delta) if s >= pos + del_len else (s, e) for s, e in ranges]
    log.append({'op': 'edit', 'pos': pos, 'before': text[pos:pos + del_len], 'after': ins})
    return (new, moved)

def _typo_site(rng, text, ranges):
    kinds = ['swap', 'double', 'drop', 'unapos', 'loweri', 'nospace']
    rng.shuffle(kinds)
    for kind in kinds:
        cands = []
        if kind == 'swap':
            cands = [i for i in range(len(text) - 1) if text[i].isalpha() and text[i + 1].isalpha() and (text[i] != text[i + 1]) and _outside(i, 2, ranges)]
            if cands:
                i = rng.choice(cands)
                return (i, 2, text[i + 1] + text[i])
        elif kind == 'double':
            cands = [i for i in range(len(text)) if text[i].isalpha() and _outside(i, 1, ranges)]
            if cands:
                i = rng.choice(cands)
                return (i, 1, text[i] * 2)
        elif kind == 'drop':
            cands = [i for i in range(len(text)) if text[i].isalpha() and _outside(i, 1, ranges)]
            if cands:
                i = rng.choice(cands)
                return (i, 1, '')
        elif kind == 'unapos':
            cands = [i for i in range(len(text)) if text[i] == "'" and _outside(i, 1, ranges)]
            if cands:
                i = rng.choice(cands)
                return (i, 1, '')
        elif kind == 'loweri':
            cands = [i for i in range(len(text)) if text[i] == 'I' and _outside(i, 1, ranges)]
            if cands:
                i = rng.choice(cands)
                return (i, 1, 'i')
        elif kind == 'nospace':
            cands = [i for i in range(1, len(text) - 1) if text[i] == ' ' and text[i - 1] == '.' and _outside(i, 1, ranges)]
            if cands:
                i = rng.choice(cands)
                return (i, 1, '')
    return None

def _curl_doubles(text: str) -> str:
    out, open_q = ([], True)
    for ch in text:
        if ch == '"':
            out.append('“' if open_q else '”')
            open_q = not open_q
        else:
            out.append(ch)
    return ''.join(out)

def noise_post(post_text: str, evidence: list[str], seed_key: str, drops_apostrophes: bool=False, lowercase_i: bool=False) -> dict | None:
    rng = random.Random(seed_key)
    ranges = _claim_ranges(post_text, evidence)
    if ranges is None:
        return None
    text = post_text
    log: list = []
    applied = []
    text, ranges, case_op = _case_pass(text, ranges, lowercase_i, f'{seed_key}|case', log)
    if case_op:
        applied.append(case_op)
    text, ranges, contract_op = _contract_pass(text, ranges, drops_apostrophes, f'{seed_key}|contract', log)
    if contract_op:
        applied.append(contract_op)
    if rng.random() < SC.NOISE['typos']:
        for _ in range(rng.randint(1, 2)):
            site = _typo_site(rng, text, ranges)
            if site:
                pos, dl, ins = site
                text, ranges = _apply(text, ranges, pos, dl, ins, log)
        applied.append('typos')
    if rng.random() < SC.NOISE['amp_entity']:
        amp = [i for i in range(len(text)) if text[i] == '&' and _outside(i, 1, ranges)]
        if amp:
            i = rng.choice(amp)
            text, ranges = _apply(text, ranges, i, 1, '&amp;', log)
        else:
            sp = [i for i in range(len(text) - 5) if text[i:i + 5] == ' and ' and _outside(i, 5, ranges)]
            if sp:
                i = rng.choice(sp)
                text, ranges = _apply(text, ranges, i, 5, ' &amp; ', log)
        applied.append('amp')
    if rng.random() < SC.NOISE['emoji']:
        em = rng.choice(SC.EMOJI_CENSUS)
        text, ranges = _apply(text, ranges, len(text), 0, ' ' + em, log)
        applied.append('emoji')
    curled = False
    if rng.random() < SC.NOISE['curly_doubles']:
        text = _curl_doubles(text)
        applied.append('curly_doubles')
    if rng.random() < SC.NOISE['curly_apostrophe_given_apos'] and "'" in text:
        text = text.replace("'", '’')
        applied.append('curly_apostrophe')
        curled = True
    nc = align.NormCache(text)
    new_evidence = []
    for s, e in ranges:
        sliced = text[s:e]
        if align.align_generated(sliced, nc) is None:
            return None
        if ';' in sliced or '\n' in sliced:
            return None
        if len(textnorm.scorer_norm(sliced).split()) > SC.SPAN_TOKENS_MAX:
            return None
        new_evidence.append(sliced)
    return {'post': text, 'evidence': new_evidence, 'applied_ops': applied, 'ops_log': log, 'curled_apostrophe': curled}
_STYLE_KEYS = ('contracted', 'dropped', 'spelled', 'apos', 'curly', 'uppercase', 'cap_i')

def noise_candidate(candidate: dict, drops_apostrophes: bool=False, lowercase_i: bool=False) -> tuple[dict | None, dict]:
    out_posts, prov_posts = ([], [])
    style_counts = {k: 0 for k in _STYLE_KEYS}
    for p in candidate['posts']:
        seed_key = f"{SC.SYNTH_SEED}|noise|{candidate['su_id']}|{p['pos']}"
        r = noise_post(p['post_text'], p['evidence'], seed_key, drops_apostrophes, lowercase_i)
        if r is None:
            return (None, {'reason': 'noise_span_lost', 'pos': p['pos']})
        for k, v in style_flags(r['post']).items():
            style_counts[k] += int(v)
        out_posts.append({**p, 'post_text': r['post'], 'evidence': r['evidence'], 'pre_noise_text': p['post_text']})
        prov_posts.append({'pos': p['pos'], 'seed_key': seed_key, 'applied_ops': r['applied_ops'], 'ops_log': r['ops_log']})
    noised = {**candidate, 'posts': out_posts, 'noise_prov': prov_posts}
    return (noised, {**style_counts, 'n_posts': len(out_posts)})

def noise_all(candidates: list[dict], drops_by_bundle: dict | None=None, lowercase_by_bundle: dict | None=None) -> tuple[list[dict], list[dict], dict]:
    out, rejects = ([], [])
    counts = {k: 0 for k in _STYLE_KEYS}
    total = 0
    for c in candidates:
        drops = bool((drops_by_bundle or {}).get(c['bundle_id'], False))
        lower = bool((lowercase_by_bundle or {}).get(c['bundle_id'], False))
        noised, stats = noise_candidate(c, drops, lower)
        if noised is None:
            rejects.append({'bundle_id': c['bundle_id'], **stats})
            continue
        out.append(noised)
        for k in _STYLE_KEYS:
            counts[k] += stats[k]
        total += stats['n_posts']
    rates = {f'{k}_post_rate': counts[k] / total if total else 0.0 for k in _STYLE_KEYS}
    return (out, rejects, {'curly_rate': rates.pop('curly_post_rate'), 'n_posts': total, **rates})
