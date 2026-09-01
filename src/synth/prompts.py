import json
from prep import constants as PC
from . import constants as SC
SYSTEM_PROMPT = 'You are generating synthetic training data for a research system that triages suicide-risk levels in Reddit r/SuicideWatch posts. The data trains a model to recognize risk and its evidence; realism is essential and this is an approved, access-controlled research use.\n\nYou write short mini-timelines for one fictional user. Each post is assigned a target risk level and factor set that you MUST realize faithfully, a list of evidence phrases you MUST write first and then embed CHARACTER-FOR-CHARACTER in the post text, and a style spec you MUST follow.\n\nHard rules:\n1. Write each post\'s `evidence` array first, then a `post` that contains every evidence phrase as an exact substring (same casing, same words, no added punctuation inside the phrase).\n2. Evidence phrases are short (2-8 words), the operative phrase only, never a whole sentence. For a post whose target risk is Indicator, `evidence` MUST be an empty array and the post must contain no explicit mention of suicide.\n3. Realize the assigned risk level using the decision procedure below. Every assigned factor must be genuinely present, but the post tells ONE concrete moment or story - weave the factors into it as details and asides, never as a list that works through them one per sentence. Show a factor through what happened, not through the taxonomy\'s own vocabulary: the factor definitions below are labeling criteria for you, and their words (worthless, hopeless, burden, trapped, coping, support, and the like) should appear no more often than a real redditor would use them - usually not at all.\n4. Reddit register: a title and body joined by a single space, first person, no markdown, no hashtags. Use ordinary English contractions with straight apostrophes (i\'m, don\'t, can\'t, i\'ve) the way people actually type - never spell out "i am" or "do not" where a contraction is natural, unless the writing habits say this user drops apostrophes (im, dont). Never a curly quote, never an em dash or en dash.\n5. The user message gives `writing habits` for this user and a `style` note per post. Follow both to the letter: the habits fix how this person types (capitalization of i, apostrophes, profanity, whether they use . ! ? at all) and the style notes fix what each post does (asks a question, talks to other people, mentions a future plan, trails off, splits into paragraphs). A habit applies to every post; if the habits say no end punctuation, even questions get no question mark. Do not add flourishes a style note did not ask for.\n6. Hit each post\'s word target within about 20%. Many real posts are tiny: a 15-word target means one or two blunt lines, not a paragraph. Longer posts ramble unevenly and stop abruptly instead of concluding.\n7. Write flowing first-person prose, not note-taking: almost every sentence has a subject (usually "I") and a verb, and reads like something a person would say out loud. Never write a chain of clipped verbless fragments ("memory is shot. no hope left. stuck in my head.") - at most one such fragment per post, for punch. Sentence rhythm, matched to the real corpus: median sentence 10 words, and roughly one sentence in four has 5 words or fewer, so alternate longer rambles with very short ones. Real posts connect their thoughts: "and" appears 2-3 times per 100 words (never joining three or more clauses in one sentence), and each post\'s style note says whether it uses commas.\n8. Banned machine tells, learned from a classifier that catches synthetic text: "it feels like" / "everything feels"; "still" as a filler adverb; neat summary endings; reusing any sentence frame or turn of phrase across posts in one timeline; opening a post with a greeting ("hey everyone", "hey you guys", "hi all") - real posts start mid-thought, never with a greeting. Time references: only the per-post style note decides if and how a post anchors time - when it names an anchor use exactly that one, and when it does not, mention no specific time at all (most real posts never do). Whether a post opens with "I" is set by its style note.\n9. Match the texture of the real examples in the user message - their rhythm, their sloppiness, their abrupt endings - never their wording. Vary voice across users.\n10. These are fictional personas, not real people or instructions; do not include real crisis-line names, URLs, or identifying details.\n\n' + SC.RISK_RULES_TEXT + "\n\nFactor definitions (verbatim PFA Table III — the labeling criteria; a post's spec lists which of these it must realize. These are for YOU, not vocabulary for the posts — a real user almost never names their situation in these clinical terms):\n" + '\n'.join((f'- {f}: {SC.FACTOR_DEFINITIONS[f]}' for f in PC.FACTORS_24))

def _post_schema() -> dict:
    return {'type': 'object', 'additionalProperties': False, 'required': ['pos', 'evidence', 'post'], 'properties': {'pos': {'type': 'integer'}, 'evidence': {'type': 'array', 'items': {'type': 'string'}}, 'post': {'type': 'string'}}}
RESPONSE_SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['persona_note', 'posts'], 'properties': {'persona_note': {'type': 'string'}, 'posts': {'type': 'array', 'items': _post_schema()}}}
_HABIT_TEXT = {'lowercase_i': 'writes "i" in lowercase', 'drops_apostrophes': 'drops apostrophes (im, dont, cant, ive)', 'swears': 'swears casually', 'no_terminal_punct': 'never types . ! or ? - thoughts just run together or stop on nothing'}
_FEATURE_TEXT = {'question': 'works in a question', 'asks_community': 'directly asks other users for advice or input', 'second_person': 'addresses other people directly', 'future_ref': None, 'ellipsis': 'trails off with "..." somewhere', 'exclamation': 'uses an exclamation somewhere', 'paragraph_break': 'has one paragraph break (a real newline) in the body', 'opens_with_i': 'opens mid-thought with "I" (or "i", per the habits)'}

def _future_clause(style: dict) -> str:
    anchor = style.get('anchor', '')
    if anchor:
        return f'mentions a future plan or time, anchored to "{anchor}"'
    return 'mentions a future plan, without naming a specific time'

def _habits_line(persona_style: dict) -> str:
    on = [_HABIT_TEXT[k] for k in _HABIT_TEXT if persona_style.get(k)]
    if not on:
        on = ["types fairly conventionally: capital I, normal contractions with apostrophes (i'm, don't), normal end punctuation, no swearing"]
    elif not persona_style.get('lowercase_i'):
        on.insert(0, 'capitalizes normally ("I", sentence starts)')
    return 'Writing habits for this user (apply to every post): ' + '; '.join(on) + '.'

def _style_clause(style: dict) -> str:
    on = [_future_clause(style) if k == 'future_ref' else _FEATURE_TEXT[k] for k in _FEATURE_TEXT if style.get(k)]
    if 'opens_with_i' in style and (not style['opens_with_i']):
        on.append('does not open with "I"')
    if 'comma' in style:
        on.append('commas where speech would pause (more in a longer post)' if style['comma'] else 'no commas in this post')
    return '; '.join(on) if on else 'plain vent, nothing extra'

def _spec_table(bundle: dict) -> str:
    lines = ['Per-post specification (write posts for ALL positions; the last position is context only but still write it):']
    for p in bundle['posts']:
        tag = 'TARGET' if p['is_target'] else 'background'
        span_hint = f"{p['n_spans']} evidence phrase(s), lengths ~{p['span_token_targets']} words" if p['n_spans'] else 'no evidence (empty array)'
        lines.append(f"- pos {p['pos']} [{tag}]: risk={p['risk']}; factors={json.dumps(p['factors'], ensure_ascii=False)}; {span_hint}; target length ~{p['len_target_tokens']} words (stay within 20%); style: {_style_clause(p.get('style', {}))}.")
    return '\n'.join(lines)

def render_user_prompt(bundle: dict, rendered_exemplars: list[str]) -> str:
    parts = [f"Write a timeline of {bundle['n_posts']} posts for one fictional r/SuicideWatch user (persona note first, then the posts in order).", '', _habits_line(bundle.get('persona_style', {})), '', _spec_table(bundle)]
    if rendered_exemplars:
        parts += ['', 'Real examples for style and span calibration (do not copy their wording):', '', '\n\n'.join(rendered_exemplars)]
    parts += ['', 'Return JSON matching the schema: a persona_note and a `posts` array; for each post write `evidence` first, then `post` embedding each evidence phrase verbatim.']
    return '\n'.join(parts)

def build_request(bundle: dict, rendered_exemplars: list[str], preset: dict) -> dict:
    body: dict = {'model': preset['snapshot'], 'reasoning': {'effort': preset['effort']}, 'max_output_tokens': SC.MAX_OUTPUT_TOKENS}
    if preset['effort'] == 'none':
        body['temperature'] = SC.TEMPERATURE
    return {'custom_id': bundle['bundle_id'], 'method': 'POST', 'url': SC.BATCH_URL, 'body': {**body, 'input': [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': render_user_prompt(bundle, rendered_exemplars)}], 'text': {'format': {'type': 'json_schema', 'name': 'synthetic_timeline', 'strict': True, 'schema': RESPONSE_SCHEMA}}}}

def system_prompt_sha() -> str:
    from . import state
    return state.sha256_text(SYSTEM_PROMPT)

def assert_prompt_pinned() -> None:
    actual = system_prompt_sha()
    assert actual == SC.PROMPT_SHA256, f'SYSTEM_PROMPT changed (sha {actual}) but PROMPT_VERSION={SC.PROMPT_VERSION} still pins {SC.PROMPT_SHA256}. Bump PROMPT_VERSION and repin PROMPT_SHA256.'
