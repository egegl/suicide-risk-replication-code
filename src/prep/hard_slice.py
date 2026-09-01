import json
import re
from pathlib import Path
CURATED = {'P01890': 'aborted_attempt', 'P00732': 'nssi_relapse', 'P01756': 'again_future', 'P01454': 'recovery_framing', 'P00576': 'recovery_framing', 'P02097': 'passing_attempt_mention', 'P02354': 'method_owned', 'P01845': 'method_question', 'P02397': 'method_question', 'P00663': 'means_acquisition', 'P01227': 'means_acquisition', 'P00750': 'set_date', 'P00123': 'set_date', 'P02149': 'location_scouting', 'P00758': 'farewell', 'P00292': 'ongoing_self_harm', 'P01303': 'passive_wish', 'P02251': 'no_access_counterfactual', 'P01612': 'external_agent', 'P00716': 'trailing_negation', 'P00759': 'dup_conflict_farewell', 'P00761': 'dup_conflict_farewell', 'P00735': 'dup_conflict_selfharm', 'P00736': 'dup_conflict_selfharm', 'P00737': 'dup_conflict_selfharm', 'P00738': 'dup_conflict_selfharm', 'P01334': 'near_dup_single_dot', 'P01335': 'near_dup_single_dot', 'P01739': 'user_context_required', 'P00866': 'gold_overcall', 'P01641': 'sleep_forever_coinflip', 'P02014': 'sleep_forever_coinflip'}
MINE_PATTERNS = {'trailing_negation': re.compile("\\b(but|though|although)\\b[^.!?]{0,60}\\b(don'?t|do not|dont|not) (really )?(want|wanna|going) to (die|do it|kill myself)\\b", re.I), 'method_question': re.compile('\\bhow (much|many|long|do|would|effective)\\b.{0,60}\\b(die|kill|od|overdose|lethal|rope|pills|hang)\\b|\\benough to (die|kill me|od)\\b', re.I), 'again_future': re.compile('\\b(gonna|going to|planning to|will)\\b.{0,30}\\b(overdose|od|try|attempt|cut)\\b.{0,15}\\bagain\\b', re.I), 'no_access_counterfactual': re.compile('\\bif (only )?i (just )?had (a|an|the|access)\\b.{0,40}\\b(gun|pills|rope|means|way)\\b|\\bwish i had (a|the) (gun|pills|rope|means)\\b', re.I), 'external_agent': re.compile('\\b(someone|somebody|god|a (car|truck|bus))\\b.{0,30}\\b(kill|shoot|stab|hit|run over) me\\b|\\bkills? me in my sleep\\b', re.I), 'recovery_framing': re.compile('\\b(kept me from|stopped me from|since my (last )?attempt|used to (want to die|be suicidal)|my attempt (last|a) (year|month))\\b', re.I)}
MINE_CAP = 20

def build(train, out_dir: Path) -> dict:
    ids = set(train['row_id'])
    missing = sorted(set(CURATED) - ids)
    assert not missing, f'curated hard-slice ids missing from train: {missing}'
    mined: dict[str, list[str]] = {}
    for tag, rx in MINE_PATTERNS.items():
        hits = [r.row_id for r in train.itertuples() if r.row_id not in CURATED and rx.search(r.post_raw)]
        mined[tag] = hits[:MINE_CAP]
    payload = {'curated': CURATED, 'mined_review_candidates': mined, 'note': 'curated = verified GUIDEBOOK §4.2/§4.3 exemplars (score against these); mined = unreviewed regex candidates (promote by hand).'}
    (out_dir / 'hard_slice.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload
