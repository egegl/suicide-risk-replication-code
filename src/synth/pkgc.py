"""Generated training data for the ICHI 2027 paper (experiment E2), plus its writer pilot.

One case per generated post, built only from the folds other than the held-out one:

- labels (risk + full factor set) and length are copied from one real carrier
  post, so the text has room for its labels;
- the new post is inserted into a real host user's timeline; the writer sees
  the host's neighbouring posts, and the labeler sees the same window the
  trained model would;
- the prompt is short on style and precise on the label contract: no surface
  rules, no example words, self-harm without an attempt is Behavior,
  protective factors carry their direction;
- the writer returns prose only; evidence comes from the fold's real-only model.

Pilot (supplement S3, exploratory): 300 cases (50 per rare factor) for held-out
fold 0, read by every candidate writer. Full dose (E2): fold-local cases for each
held-out fold, written by Claude Sonnet 5.5 (scripts/pkgc_claude.py) and
Gemma-4-31B-IT (scripts/pkgc_local.py). CPU steps run through scripts/e2_data.py.
"""
from __future__ import annotations

import difflib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from prep import constants as PC
from prep import context
from . import bundles
from . import constants as SC
from prep import textnorm
from . import filters, state

PROGRAM = "synthetic-pkgc-pilot"
PROMPT_VERSION = "pkgc-v2"
SEED = 20261015
POOL = 0                      # held-out fold; every input comes from folds != 0
CASES_PER_FACTOR = 50
N_EARLIER, N_LATER = PC.CTX_N_PREV, PC.CTX_N_NEXT   # the trained model's window
REFERENCE_CHARS = 1200        # writer sees fuller neighbours than the model's 300
MAX_HOST_USES = 3
POST_WORDS_REJECT = 700
POST_WORDS_MIN, POST_WORDS_MAX = 15, 350   # target-length clamp
REFERENCE_OVERLAP_NGRAM = 6               # copy check against the shown neighbours
REFERENCE_SEQUENCE_REJECT = 0.55
ARMS = ("haiku", "sonnet", "qwen", "hemmingway", "qwenflash", "gemma12", "gemma31")
ROOTS = {"pkgc-v1": Path("data/synthetic_pkgc/pilot"),
         "pkgc-v2": Path("data/synthetic_pkgc/pilot_v2"),
         "pkgc-v2-subtle": Path("data/synthetic_pkgc/pilot_v2_subtle")}
ROOT = ROOTS[PROMPT_VERSION]

_SYSTEM_HEAD = """\
You write one Reddit post for a research dataset that trains a suicide-risk classifier. You are shown real posts by one r/SuicideWatch user, in order, with a marked gap. Write the post that fills the gap: the same person at a new moment, not a rewrite of their other posts.

"""
SYSTEM_PROMPTS = {
    # v1 (pilot 2026-10-15): its "with no plan" clause was narrated back in 41%
    # of Sonnet Ideation posts ("No plan, I'm not doing anything").
    "pkgc-v1": _SYSTEM_HEAD + """\
A careful annotator must be able to read the assigned annotation off your post. Risk levels:
- Indicator: no explicit mention of suicide or of wanting to die.
- Ideation: explicit suicidal thoughts or a wish to die, with no plan, method, or self-harm.
- Behavior: suicidal thoughts with a plan, a method, preparation, or self-harm. Self-harm without a suicide attempt is Behavior.
- Attempt: the author describes their own actual suicide attempt, recent or past, including one they stopped partway.

Show the risk level and every assigned factor through what the person says and what happened; never name the labels. Return only the text of the post.""",
    # v2: each level says what the post shows; boundaries are stated as what
    # would move a post up a level, and absences are never to be written out.
    "pkgc-v2": _SYSTEM_HEAD + """\
A careful annotator must be able to read the assigned annotation off your post. What each risk level shows:
- Indicator: distress, without any explicit mention of suicide or of wanting to die.
- Ideation: explicit thoughts of suicide or of wanting to die. A plan, a method, or self-harm would make it Behavior.
- Behavior: explicit suicidal thoughts together with a plan, a method, preparation, or self-harm, including self-harm without a suicide attempt.
- Attempt: the author's own actual suicide attempt, recent or past, including one they stopped partway.

Write only what this person would actually say. Show the risk level and every assigned factor through what they say and what happened. Never name the labels, explain them, or state what is not happening. Return only the text of the post.""",
}
# C-subtle (PAPER_DESIGN_C.md): v2 plus one paragraph. Generated posts were
# prototypes in the pilot (labeler target recall 0.64-0.77 vs ~0.30 on real
# rare-factor posts); this asks for factors to surface the way real posts show them.
SUBTLE_PARAGRAPH = (
    "In real posts these things usually come up indirectly: mentioned in passing, "
    "implied by a detail, or buried in a longer story. Let each factor surface that "
    "way rather than making it the point of the post.")
SYSTEM_PROMPTS["pkgc-v2-subtle"] = SYSTEM_PROMPTS["pkgc-v2"].replace(
    " Return only the text of the post.",
    "\n\n" + SUBTLE_PARAGRAPH + "\n\nReturn only the text of the post.")
assert SYSTEM_PROMPTS["pkgc-v2-subtle"] != SYSTEM_PROMPTS["pkgc-v2"]
SYSTEM_PROMPT = SYSTEM_PROMPTS[PROMPT_VERSION]

# PFA Table III wording, except where it gives no direction (audit A4) or is vague.
PROTECTIVE = ("coping strategy", "psychological capital", "social support",
              "sense of responsibility", "meaning in life")
DEFINITION_OVERRIDES = {
    "pkgc-v1": {
        "meaning in life": "Reasons to live, purpose, or something that makes life feel worth living",
        "psychological capital": "Positive inner resources such as hope, resilience, or confidence",
        "cognitive deficits": "Difficulty with cognitive abilities such as memory, focus, or thinking clearly",
    },
    # v2: no example words anywhere ("burden" from low self-esteem's e.g. list
    # reached 25% of v1 Sonnet posts; "focus" became "I read the same paragraph").
    "pkgc-v2": {
        "meaning in life": "Having reasons to live or a sense of purpose",
        "psychological capital": "A positive inner state the person can draw on",
        "cognitive deficits": "Difficulty with cognitive abilities",
    },
}
DEFINITION_OVERRIDES["pkgc-v2-subtle"] = DEFINITION_OVERRIDES["pkgc-v2"]
_EXAMPLES = re.compile(r"\s*\(e\.g\.[^)]*\)")


def definition(factor: str, version: str = PROMPT_VERSION) -> str:
    text = DEFINITION_OVERRIDES[version].get(factor, SC.FACTOR_DEFINITIONS[factor])
    if version != "pkgc-v1":
        text = _EXAMPLES.sub("", text)
    return f"{text} (protective)" if factor in PROTECTIVE else text


# ---------------------------------------------------------------- text helpers
# Copied unchanged from the competition-era generator modules.

def _clamp_words(n: int) -> int:
    return max(POST_WORDS_MIN, min(int(n), POST_WORDS_MAX))


def _carrier(rng: random.Random, rows: list[dict], factor: str, serial: int) -> dict:
    carriers = [r for r in rows if factor in r["factors"]]
    assert carriers, factor
    low = [r for r in carriers if r["risk"] in ("Indicator", "Ideation")]
    high = [r for r in carriers if r["risk"] in ("Behavior", "Attempt")]
    # Stress both severity bands when the observed carrier set permits it.
    band = high if serial % 2 == 0 and high else low if low else carriers
    return rng.choice(band)


def _sequence_ratio(left: str, right: str) -> float:
    return difflib.SequenceMatcher(
        None, textnorm.normalize(left), textnorm.normalize(right),
        autojunk=False,
    ).ratio()


def _has_reference_ngram(post: str, references: list[str]) -> bool:
    post_tokens = textnorm.normalize(post).split()
    post_ngrams = {
        tuple(post_tokens[i:i + REFERENCE_OVERLAP_NGRAM])
        for i in range(len(post_tokens) - REFERENCE_OVERLAP_NGRAM + 1)
    }
    for reference in references:
        tokens = textnorm.normalize(reference).split()
        for i in range(len(tokens) - REFERENCE_OVERLAP_NGRAM + 1):
            if tuple(tokens[i:i + REFERENCE_OVERLAP_NGRAM]) in post_ngrams:
                return True
    return False


_CAP_I = re.compile(r"(?<![A-Za-z])I(?![A-Za-z])")
_EMOJI = re.compile(
    "[\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF\u2600-\u27BF]",
    flags=re.UNICODE,
)


def _style(texts: list[str]) -> dict:
    n = len(texts)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "any_uppercase": round(sum(any(c.isupper() for c in t) for t in texts) / n, 4),
        "capital_i": round(sum(bool(_CAP_I.search(t)) for t in texts) / n, 4),
        "emoji": round(sum(bool(_EMOJI.search(t)) for t in texts) / n, 4),
        "comma": round(sum("," in t for t in texts) / n, 4),
        "question": round(sum("?" in t for t in texts) / n, 4),
        "mean_words": round(sum(len(t.split()) for t in texts) / n, 2),
    }


def _top_openings(texts: list[str], n_words: int, limit: int = 15) -> list[dict]:
    counts = Counter(
        " ".join(text.lower().split()[:n_words])
        for text in texts if text.strip()
    )
    return [
        {"opening": opening, "count": count}
        for opening, count in counts.most_common(limit)
        if count > 1
    ]


# ---------------------------------------------------------------- cases

def _timelines(processed: Path, pool: int = POOL) -> dict[str, list[dict]]:
    posts = pd.read_parquet(processed / "posts.parquet")
    posts = posts[(posts["split"] == "train") & (posts["fold"] != pool)]
    out: dict[str, list[dict]] = {}
    for user, g in posts.sort_values(["anon_user_id", "post_id"]).groupby("anon_user_id"):
        out[user] = [{"row_id": r.row_id, "text": r.post_raw} for r in g.itertuples()]
    return out


def build_cases(processed: Path, pool: int = POOL, n_cases: int | None = None) -> list[dict]:
    """Pilot (n_cases=None): 50 cases per factor for fold 0, carriers alternating
    severity bands, byte-identical to the frozen pilot. Full dose (n_cases set):
    fold-local cases for held-out fold `pool`, carriers drawn at their natural
    risk mix, hosts reused as often as the dose needs, ids "K<fold>C<n>"."""
    full = n_cases is not None
    rows = [r for r in bundles.load_rows(processed)["rows"] if r["fold"] != pool]
    timelines = _timelines(processed, pool)
    hosts = sorted(u for u, t in timelines.items() if len(t) >= 3)
    per_factor = -(-n_cases // len(SC.RARE_FACTORS)) if full else CASES_PER_FACTOR
    max_uses = -(-n_cases // len(hosts)) + 2 if full else MAX_HOST_USES
    ns = f"|full|pool{pool}" if full else ""
    host_uses: Counter = Counter()
    cases = []
    for serial in range(per_factor):
        order = list(SC.RARE_FACTORS)
        random.Random(f"{SEED}{ns}|order|{serial}").shuffle(order)
        for factor in order:
            rng = random.Random(f"{SEED}{ns}|case|{factor}|{serial}")
            if full:
                carrier = rng.choice([r for r in rows if factor in r["factors"]])
            else:
                carrier = _carrier(rng, rows, factor, serial)
            free = [u for u in hosts
                    if u != carrier["anon_user_id"] and host_uses[u] < max_uses]
            host = rng.choice(free)
            host_uses[host] += 1
            timeline = timelines[host]
            gap = rng.randint(1, len(timeline) - 1)   # at least one post on each side
            cases.append({
                "case_id": (f"K{pool}C{len(cases) + 1:05d}" if full else f"C{len(cases) + 1:04d}"),
                "target_factor": factor,
                "risk": carrier["risk"],
                "factors": list(carrier["factors"]),
                "approx_words": _clamp_words(carrier["n_tokens"]),
                "carrier_row_id": carrier["row_id"],
                "host_user": host,
                "gap": gap,
                "earlier_ids": [p["row_id"] for p in timeline[max(0, gap - N_EARLIER):gap]],
                "later_ids": [p["row_id"] for p in timeline[gap:gap + N_LATER]],
            })
    if full:
        return cases[:n_cases]
    assert Counter(c["target_factor"] for c in cases) == \
        Counter({f: CASES_PER_FACTOR for f in SC.RARE_FACTORS})
    return cases


def _ref(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= REFERENCE_CHARS else text[:REFERENCE_CHARS].rstrip() + " [...]"


def render_user_prompt(case: dict, text_by_id: dict[str, str],
                       version: str = PROMPT_VERSION) -> str:
    refs = [_ref(text_by_id[i]) for i in case["earlier_ids"]]
    later = [_ref(text_by_id[i]) for i in case["later_ids"]]
    lines = ["Posts by this user, oldest first:", ""]
    n = 0
    for text in refs:
        n += 1
        lines += [f"[{n}] {text}", ""]
    lines += ["[NEW POST GOES HERE]", ""]
    for text in later:
        n += 1
        lines += [f"[{n}] {text}", ""]
    lines += ["Annotation of the new post:", f"Risk: {case['risk']}", "Factors present:"]
    lines += [f"- {f}: {definition(f, version)}" for f in case["factors"]]
    lines += [f"Length: about {case['approx_words']} words."]
    return "\n".join(lines)


def text_by_id(processed: Path) -> dict[str, str]:
    posts = pd.read_parquet(processed / "posts.parquet", columns=["row_id", "post_raw"])
    return dict(zip(posts.row_id, posts.post_raw))


# ---------------------------------------------------------------- full dose (E2)
# Each held-out fold's file is real rows + generated rows filled to exactly the
# size of the repeated-real (qcontrol) file, so runs31/qcontrol stays the
# size-matched control. Cases are oversampled for rows lost to checks and spans.
FULL_ROOT = Path("data/synthetic_pkgc/full")
QCONTROL_DIR = Path("data/synthetic/arms/qcontrol")
FULL_OVERSAMPLE = 1.10
# Rows added per held-out fold by the repeated-real control (registered). In the
# competition sweep they were sized to the GPT-5.4-mini top-up arm; frozen here so
# E2 and its control can be rebuilt without that pipeline (scripts/e2_controls.py).
FULL_DOSE = (1211, 1309, 1309, 1234, 1321)


def full_root(version: str, fold: int) -> Path:
    return FULL_ROOT / version / f"fold{fold}"


def full_dose(fold: int) -> int:
    return FULL_DOSE[fold]


def prepare(processed: Path, version: str = PROMPT_VERSION, fold: int | None = None) -> dict:
    """fold=None: the 300-case fold-0 pilot. fold=k: full-dose cases for fold k."""
    system = SYSTEM_PROMPTS[version]
    if fold is None:
        root, pool, cases = ROOTS[version], POOL, build_cases(processed)
    else:
        root, pool = full_root(version, fold), fold
        cases = build_cases(processed, fold, int(full_dose(fold) * FULL_OVERSAMPLE) + 1)
    texts = text_by_id(processed)
    prompts = [{"case_id": c["case_id"], "system": system,
                "user": render_user_prompt(c, texts, version)} for c in cases]
    state.atomic_write_jsonl(root / "cases.jsonl", cases)
    state.atomic_write_jsonl(root / "prompts.jsonl", prompts)
    manifest = {
        "program": PROGRAM, "prompt_version": version, "seed": SEED,
        "pool": pool, "n_cases": len(cases), "arms": list(ARMS),
        "dose": full_dose(fold) if fold is not None else None,
        "system_prompt_sha256": state.sha256_text(system),
        "cases_sha256": state.sha256_file(root / "cases.jsonl"),
        "prompts_sha256": state.sha256_file(root / "prompts.jsonl"),
        "exploratory": True,
    }
    state.atomic_write_json(root / "manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------- outputs

# Haiku 5.5 declined 168/299 pilot cases in-text with stop_reason end_turn.
_SOFT_REFUSAL = re.compile(
    r"(can'?t|cannot|won'?t|not able to|not going to|don'?t want to) (write|produce|create|help (with|write))"
    r"|i can help in other ways|\b988\b|crisis (line|text)|call or text|findahelpline", re.I)


# "Not a plan, just the thought": Sonnet v1 41%, v2 18% of Ideation posts.
_PLAN_DISCLAIMER = re.compile(
    r"(no|not|don'?t|n't|n’t|never|without)[^.?!]{0,30}\b(plan|method|pills)", re.I)


def clean_post(text: str | None) -> str:
    text = (text or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def screen(cases: list[dict], outputs: dict[str, dict], texts: dict[str, str],
           real_index: filters.RealIndex) -> tuple[list[dict], Counter]:
    """Accepted rows (case + post) and reject reasons. No evidence gate here."""
    accepted, reasons = [], Counter()
    for case in cases:
        out = outputs.get(case["case_id"]) or {"status": "missing"}
        post = clean_post(out.get("text"))
        refs = [texts[i] for i in case["earlier_ids"] + case["later_ids"]]
        if out["status"] != "ok":
            reason = f"status:{out['status']}"
        elif not post:
            reason = "empty"
        elif _SOFT_REFUSAL.search(post):
            reason = "soft_refusal"   # refusal + crisis-line text returned as end_turn
        elif case["risk"] == "Ideation" and _PLAN_DISCLAIMER.search(post):
            reason = "plan_disclaimer"   # narrated label rule; 1.8% of real Ideation posts
        elif len(post.split()) > POST_WORDS_REJECT:
            reason = "too_long"
        elif real_index.hits(post):
            reason = "real_overlap"
        elif (_has_reference_ngram(post, refs) or max(
                _sequence_ratio(post, r) for r in refs) >= REFERENCE_SEQUENCE_REJECT):
            reason = "reference_copy"
        else:
            reason = None
        if reason:
            reasons[reason] += 1
            continue
        accepted.append({**case, "post": post})
    return accepted, reasons


DRAFTS = 2   # v2: drafts per case, one kept by within-factor diversity


def _grams4(text: str) -> set:
    w = re.findall(r"[a-z']+", text.lower().replace("’", "'"))
    return {tuple(w[i:i + 4]) for i in range(len(w) - 3)}


def match_apostrophes(post: str, host_texts: list[str]) -> str:
    """Typing-device habit, not content: curl apostrophes when the host does."""
    curly = sum(t.count("’") for t in host_texts)
    return post.replace("'", "’") if curly > sum(t.count("'") for t in host_texts) else post


def select_drafts(cases: list[dict], drafts: dict[str, list[dict]], texts: dict[str, str],
                  real_index: filters.RealIndex) -> list[dict]:
    """One output row per case. Among drafts that pass screen(), keep the one
    whose 4-grams overlap least (max Jaccard) with posts already kept for the
    same target factor; cases are visited in their fixed order. A case with no
    passing draft keeps its first draft so its reject reason is reported."""
    kept: dict[str, list[set]] = defaultdict(list)
    out = []
    for case in cases:
        options = []
        for k, d in enumerate(drafts.get(case["case_id"], [])):
            if screen([case], {case["case_id"]: d}, texts, real_index)[0]:
                g = _grams4(d["text"])
                sim = max((len(g & h) / len(g | h) for h in kept[case["target_factor"]]
                           if g | h), default=0.0)
                options.append((sim, k, d, g))
        if not options:
            first = (drafts.get(case["case_id"]) or [{"case_id": case["case_id"], "status": "missing"}])[0]
            out.append({**first, "draft": 0, "drafts_passing": 0})
            continue
        sim, k, d, g = min(options, key=lambda o: (o[0], o[1]))
        kept[case["target_factor"]].append(g)
        host = [texts[i] for i in case["earlier_ids"] + case["later_ids"]]
        out.append({**d, "text": match_apostrophes(clean_post(d["text"]), host),
                    "draft": k, "drafts_passing": len(options), "max_jaccard_kept": round(sim, 4)})
    return out


def select_draft_rows(draft_rows: list[dict], root: Path,
                      processed: Path = Path("data/processed")) -> list[dict]:
    """Draft rows carry case ids "C0001-d1"; returns one selected row per case."""
    drafts: dict[str, list[dict]] = {}
    for r in draft_rows:
        cid, k = r["case_id"].rsplit("-d", 1)
        drafts.setdefault(cid, []).append({**r, "case_id": cid, "draft_id": int(k)})
    for v in drafts.values():
        v.sort(key=lambda r: r["draft_id"])
    return select_drafts(state.read_jsonl(root / "cases.jsonl"), drafts, text_by_id(processed),
                         filters.RealIndex(filters.load_real_posts(processed)))


def labeler_input(case: dict, post: str, timelines: dict[str, list[dict]]) -> str:
    """The trained model's exact input: host timeline with the post inserted."""
    posts = [p["text"] for p in timelines[case["host_user"]]]
    posts.insert(case["gap"], post)
    text, _, _ = context.build_input(posts, case["gap"])
    return text


# ---------------------------------------------------------------- report

_SUICIDE = re.compile(r"\b(suicid\w*|kill(ing)? myself|end(ing)? (it|my life)|want(ed)? to die|wanna die)\b", re.I)
_SELF_HARM = re.compile(r"\b(cut(ting)?|self[- ]?harm\w*|burn(ed|ing)? myself|scars?)\b", re.I)
_TAXONOMY = re.compile(r"\b(worthless\w*|hopeless\w*|burden\w*|trapped|coping|cope|support system|"
                       r"self[- ]esteem|isolated|meaning(less)?|purpose)\b", re.I)


def _rate(rows, fn) -> float:
    return round(sum(bool(fn(r)) for r in rows) / len(rows), 4) if rows else 0.0


def _fidelity(accepted: list[dict], labels: dict[str, dict]) -> dict:
    rows = [(c, labels[c["case_id"]]) for c in accepted
            if c["case_id"] in labels and not labels[c["case_id"]].get("parse_failure")]
    if not rows:
        return {"n": 0, "parse_failures": sum(bool(l.get("parse_failure")) for l in labels.values())}
    tp = fp = fn = 0
    per_target: dict[str, list[int]] = defaultdict(list)
    for case, lab in rows:
        prog, pred = set(case["factors"]), set(lab["factors"])
        tp += len(prog & pred); fp += len(pred - prog); fn += len(prog - pred)
        per_target[case["target_factor"]].append(int(case["target_factor"] in pred))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(rows),
        "risk_agreement": round(sum(c["risk"] == l["risk"] for c, l in rows) / len(rows), 4),
        "risk_confusion": dict(Counter(f"{c['risk']}->{l['risk']}" for c, l in rows
                                       if c["risk"] != l["risk"]).most_common(8)),
        "target_recall": round(sum(sum(v) for v in per_target.values()) / len(rows), 4),
        "target_recall_by_factor": {f: round(sum(v) / len(v), 3) for f, v in sorted(per_target.items())},
        "factor_micro_p": round(p, 4), "factor_micro_r": round(r, 4),
        "factor_micro_f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
    }


def arm_report(arm: str, cases: list[dict], outputs: dict[str, dict], texts: dict[str, str],
               real_index, labels: dict[str, dict], processed: Path,
               realism: bool = True) -> dict:
    from . import realism_headroom as rh
    accepted, reasons = screen(cases, outputs, texts, real_index)
    posts = [c["post"] for c in accepted]
    ratio = [len(c["post"].split()) / c["approx_words"] for c in accepted]
    rep = {
        "arm": arm,
        "n_cases": len(cases),
        "status": dict(Counter(o.get("status", "missing") for o in outputs.values())),
        "accepted": len(accepted),
        "rejects": dict(reasons),
        "length_ratio_median": round(statistics.median(ratio), 3) if ratio else None,
        "style": _style(posts),
        "top_openings_2w": _top_openings(posts, 2, limit=6),
        "indicator_with_suicide_words": _rate([c for c in accepted if c["risk"] == "Indicator"],
                                              lambda c: _SUICIDE.search(c["post"])),
        "ideation_with_self_harm": _rate([c for c in accepted if c["risk"] == "Ideation"],
                                         lambda c: _SELF_HARM.search(c["post"])),
        "taxonomy_vocabulary": _rate(accepted, lambda c: _TAXONOMY.search(c["post"])),
        "label_fidelity": _fidelity(accepted, labels),
        "usage": _usage(outputs),
    }
    if realism and len(accepted) >= 50:
        records = rh.case_records(accepted, arm)
        rep["realism_leakfree"] = rh.leakfree_auc(records, processed)
    return rep


def _usage(outputs: dict[str, dict]) -> dict:
    tot: Counter = Counter()
    for o in outputs.values():
        tot.update({k: v for k, v in (o.get("usage") or {}).items() if isinstance(v, int)})
    return dict(tot)


def real_style(processed: Path) -> dict:
    posts = pd.read_parquet(processed / "posts.parquet")
    posts = posts[(posts["split"] == "train") & (posts["fold"] != POOL)]
    texts = posts["post_raw"].tolist()
    return {"style": _style(texts), "top_openings_2w": _top_openings(texts, 2, limit=6),
            "taxonomy_vocabulary": _rate(texts, _TAXONOMY.search)}


def labeler_real_baseline(processed: Path, oofs: tuple = (Path("runs31/baseline-f0-s1/oof.jsonl"),)) -> dict:
    """The labeler (or each fold's labeler) on its real held-out posts that carry
    a rare factor: the rate a generated post is compared against. With all five
    runs31/baseline-f*-s1 files this pools ~140 posts (fold 4 selected step 120)."""
    missing = [str(o) for o in oofs if not Path(o).exists()]
    if missing:
        return {"n": 0, "missing": missing}
    preds = {r["row_id"]: r for o in oofs for r in state.read_jsonl(o)}
    gold = pd.read_parquet(processed / "gold.parquet")
    cases = []
    for g in gold[gold["row_id"].isin(preds)].itertuples():
        factors = set(g.factors_set)
        for f in SC.RARE_FACTORS:
            if f in factors:
                cases.append({"case_id": f"{g.row_id}:{f}", "target_factor": f,
                              "risk": g.risk_train, "factors": sorted(factors)})
    labels = {c["case_id"]: preds[c["case_id"].split(":")[0]] for c in cases}
    return _fidelity(cases, labels)


def report(processed: Path, root: Path = ROOT, realism: bool = True) -> dict:
    cases = state.read_jsonl(root / "cases.jsonl")
    texts = text_by_id(processed)
    real_index = filters.RealIndex(filters.load_real_posts(processed))
    out = {"manifest": json.loads((root / "manifest.json").read_text()),
           "real_reference": real_style(processed),
           "labeler_real_fold0": labeler_real_baseline(processed),
           "labeler_real_all_folds": labeler_real_baseline(processed, tuple(
               Path(f"runs31/baseline-f{k}-s1/oof.jsonl") for k in range(5))),
           "arms": {}}
    for arm in ARMS:
        path = root / "outputs" / f"{arm}.jsonl"
        if not path.exists():
            continue
        outputs = {o["case_id"]: o for o in state.read_jsonl(path)}
        lpath = root / "labels" / f"{arm}.jsonl"
        labels = {l["case_id"]: l for l in state.read_jsonl(lpath)} if lpath.exists() else {}
        out["arms"][arm] = arm_report(arm, cases, outputs, texts, real_index, labels,
                                      processed, realism=realism)
    state.atomic_write_json(root / "report.json", out)
    return out


# ---------------------------------------------------------------- assembly (E2)
ARMS_DIR = Path("data/synthetic_pkgc/arms")


def _span_ranges(post: str, spans: list[str]) -> list[tuple[int, int]] | None:
    """First non-overlapping occurrence of each span, in post order; None if a
    span is not a verbatim substring (the row is then unusable)."""
    claimed: list[tuple[int, int]] = []
    for sp in spans:
        start = 0
        while True:
            i = post.find(sp, start)
            if i == -1:
                return None
            j = i + len(sp)
            if not any(i < e and s < j for s, e in claimed):
                claimed.append((i, j))
                break
            start = i + 1
    return sorted(claimed)


def assemble_fold(arm: str, fold: int, version: str = "pkgc-v2",
                  processed: Path = Path("data/processed")) -> tuple[list[dict], dict]:
    """Real fold-k training rows + exactly full_dose(k) generated rows (the
    first usable ones in case order), interleaved by one seeded shuffle, so the
    file matches the runs31 repeated-real file row for row. A generated row is
    usable if it passes screen() and, unless Indicator, the fold-k labeler gave
    it at least one verbatim evidence span. Labels are the programmed ones;
    only the evidence comes from the labeler."""
    from prep import target as prep_target
    root = full_root(version, fold)
    cases = state.read_jsonl(root / "cases.jsonl")
    outs = {o["case_id"]: o for o in state.read_jsonl(root / "outputs" / f"{arm}.jsonl")}
    labels = {l["case_id"]: l for l in state.read_jsonl(root / "labels" / f"{arm}.jsonl")}
    texts = text_by_id(processed)
    accepted, reasons = screen(cases, outs, texts, filters.RealIndex(filters.load_real_posts(processed)))
    timelines = _timelines(processed, fold)
    fold_of = dict(pd.read_parquet(processed / "posts.parquet", columns=["row_id", "fold"]).itertuples(index=False))
    dose, synth = full_dose(fold), []
    for c in accepted:
        if len(synth) == dose:
            break
        post, lab = c["post"], labels.get(c["case_id"], {})
        ranges = [] if c["risk"] == "Indicator" else _span_ranges(post, lab.get("spans") or [])
        if lab.get("parse_failure") or ranges is None or (c["risk"] != "Indicator" and not ranges):
            reasons["no_evidence_span"] += 1
            continue
        posts = [p["text"] for p in timelines[c["host_user"]]]
        posts.insert(c["gap"], post)
        input_text, _, is_trunc = context.build_input(posts, c["gap"], cover=ranges or None)
        sources = [c["carrier_row_id"], *c["earlier_ids"], *c["later_ids"]]
        assert all(fold_of[s] != fold for s in sources), (c["case_id"], fold)
        synth.append({
            "row_id": f"{arm[:2].upper()}{c['case_id']}",
            "input": input_text, "is_truncated": is_trunc, "fold": -1, "weight": 1.0,
            "target": prep_target.render_target(
                c["risk"], [post[s:e] for s, e in ranges],
                sorted(set(c["factors"]), key=PC.FACTOR_INDEX.__getitem__), PC.TARGET_SCHEMA),
            # exemplar_row_ids feeds the trainer's fold-leakage re-assert (train.data)
            "synth": {"exemplar_row_ids": sources, "pool_fold": fold, "case_id": c["case_id"],
                      "arm": arm, "prompt_version": version, "target_factor": c["target_factor"]},
        })
    if len(synth) < dose:
        raise SystemExit(f"{arm} fold{fold}: only {len(synth)} usable rows for dose {dose}")
    real = state.read_jsonl(processed / "sft" / f"train_fold{fold}.jsonl")
    merged = real + synth
    random.Random(f"{SEED}|assemble|{arm}|{fold}").shuffle(merged)
    n_q = sum(1 for _ in open(QCONTROL_DIR / f"train_fold{fold}.jsonl"))
    assert len(merged) == n_q, (len(merged), n_q)
    stats = {"real": len(real), "synthetic": len(synth), "total": len(merged),
             "qcontrol_total": n_q, "rejects_before_dose": dict(reasons),
             "rare_positives": {f: sum(f in json.loads(r["target"])["factors"] and
                                       json.loads(r["target"])["factors"][f] for r in synth)
                                for f in SC.RARE_FACTORS}}
    return merged, stats


def assemble(arm: str, version: str = "pkgc-v2") -> dict:
    out = ARMS_DIR / f"{arm}_{version}"
    manifest = {"arm": arm, "prompt_version": version, "seed": SEED, "folds": {}}
    for k in range(5):
        rows, stats = assemble_fold(arm, k, version)
        state.atomic_write_jsonl(out / f"train_fold{k}.jsonl", rows)
        manifest["folds"][k] = {**stats, "sha256": state.sha256_file(out / f"train_fold{k}.jsonl")}
        print(arm, f"fold{k}", stats["synthetic"], "+", stats["real"], "=", stats["total"],
              "| rare positives", stats["rare_positives"])
    state.atomic_write_json(out / "manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------- E2 specs (registered)
E2_RUNS_DIR = Path("runs31-pkgc")
E2_ARMS = {"pkgc_sonnet": ARMS_DIR / "sonnet_pkgc-v2", "pkgc_gemma": ARMS_DIR / "gemma31_pkgc-v2"}


def write_e2_specs(runs_dir: Path = E2_RUNS_DIR) -> list[dict]:
    """20 specs in run_experiment.py's spec shape: the runs31 baseline spec for the
    same fold and seed with only arm, run_id, train_file, its sha, and
    mean_weight changed (registered: same base model, steps, checkpoints,
    seeds). Arm-major, then fold, then seed; indices 0-19 are the train array."""
    base = {(s["fold"], s["seed"]): s for s in json.loads(Path("runs31/specs.json").read_text())
            if s["arm"] == "baseline" and s["fold"] != "full"}
    specs = []
    for arm, arm_dir in E2_ARMS.items():
        manifest = json.loads((arm_dir / "manifest.json").read_text())
        for fold in range(5):
            path = arm_dir / f"train_fold{fold}.jsonl"
            sha = state.sha256_file(path)
            assert sha == manifest["folds"][str(fold)]["sha256"], (arm, fold)
            weights = [json.loads(l)["weight"] for l in open(path)]
            for seed in (1, 2):
                spec = dict(base[(fold, seed)])
                spec.update(run_id=f"{arm}-f{fold}-s{seed}", index=len(specs), arm=arm,
                            train_file=str(path), train_file_sha=sha,
                            mean_weight=sum(weights) / len(weights))
                specs.append(spec)
    state.atomic_write_json(runs_dir / "specs.json", specs)
    return specs


# ---------------------------------------------------------------- E2 human audit (registered)
AUDIT_SEED = 20261015
AUDIT_PER_FACTOR = 4
AUDIT_DIR = Path("data/synthetic_pkgc/audit/e2_v1")


def _rank(*parts) -> str:
    return state.sha256_text("|".join(map(str, (AUDIT_SEED, *parts))))


def _training_posts(arm_dir: Path) -> dict[str, dict]:
    """case_id -> {post, risk, evidence, factors, target_factor, fold} for every
    generated row in an arm's five training files; the post is the exact focal
    text trained on."""
    out = {}
    for k in range(5):
        for line in open(arm_dir / f"train_fold{k}.jsonl"):
            r = json.loads(line)
            if "synth" not in r:
                continue
            post = r["input"].split("[FOCAL POST]\n", 1)[1].split("\n[Later post", 1)[0]
            t = json.loads(r["target"])
            out[r["synth"]["case_id"]] = {
                "post": post, "risk": t["risk"], "evidence": t["evidence"],
                "factors": [f for f, v in t["factors"].items() if v],
                "target_factor": r["synth"]["target_factor"], "fold": k}
    return out


def audit_sample(arms: dict[str, dict]) -> list[str]:
    """The registered audit sample: AUDIT_PER_FACTOR cases per rare target
    factor, the same cases for both writers (only cases trained on in both
    arms), risk mix proportional to that common pool (largest remainder),
    picked in a fixed seeded order before any annotation."""
    common = sorted(set.intersection(*(set(v) for v in arms.values())))
    first = arms["pkgc_sonnet"]
    n = AUDIT_PER_FACTOR * len(SC.RARE_FACTORS)
    mix = Counter(first[c]["risk"] for c in common)
    exact = {r: n * mix[r] / len(common) for r in PC.RISK_CLASSES}
    quota = {r: int(exact[r]) for r in PC.RISK_CLASSES}
    for r in sorted(PC.RISK_CLASSES, key=lambda r: exact[r] - quota[r], reverse=True)[:n - sum(quota.values())]:
        quota[r] += 1
    chosen = []
    for factor in SC.RARE_FACTORS:
        pool = sorted((c for c in common if first[c]["target_factor"] == factor),
                      key=lambda c: _rank("case", c))
        picked = []
        for c in pool:
            if len(picked) < AUDIT_PER_FACTOR and quota[first[c]["risk"]] > 0:
                picked.append(c)
                quota[first[c]["risk"]] -= 1
        for c in pool:   # this factor's quota-eligible cases ran out: next ranked case
            if len(picked) < AUDIT_PER_FACTOR and c not in picked:
                picked.append(c)
        chosen += picked
    return chosen


def build_audit_packet(annotators=("annotator1", "annotator2", "annotator3"),
                       tool_dir: Path = Path("audit")) -> dict:
    """packet.json + manifest.json in the synthetic-semantic-audit-v1 format the
    annotators' tool reads (risk definitions = the tool's clarified ones, factor
    definitions = PFA Table III), plus the unblinding key, which stays out of
    the package sent to annotators."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("annotate", tool_dir / "annotate.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    arms = {a: _training_posts(d) for a, d in E2_ARMS.items()}
    items = sorted(((arm, c) for c in audit_sample(arms) for arm in E2_ARMS),
                   key=lambda x: _rank("order", *x))
    cases, key = [], {}
    for i, (arm, c) in enumerate(items, 1):
        cid, rec = f"CA{i:02d}", arms[arm][c]
        cases.append({"case_id": cid, "focal_post": rec["post"],
                      "post_sha256": state.sha256_text(rec["post"])})
        key[cid] = {"arm": arm, "source_case_id": c, **{k: v for k, v in rec.items() if k != "post"}}
    packet = {
        "protocol": tool.PROTOCOL,
        "blind": ("Assigned labels, target factor, generator identity, and generation metadata are "
                  "intentionally absent. Do not open assigned_key.DO_NOT_OPEN.json until all "
                  "annotations and the consensus are complete."),
        "task": {"risk_classes": list(PC.RISK_CLASSES), "risk_definitions": tool.RISK_DEFINITIONS,
                 "factor_taxonomy": list(PC.FACTORS_24),
                 "factor_definitions": {f: SC.FACTOR_DEFINITIONS[f] for f in PC.FACTORS_24},
                 "cases": cases}}
    state.atomic_write_json(AUDIT_DIR / "packet.json", packet)
    manifest = {
        "protocol": tool.PROTOCOL, "status": "e2_audit_round_1",
        "registration": "E2 human audit (claim C1)",
        "seed": AUDIT_SEED, "per_factor": AUDIT_PER_FACTOR,
        "case_ids": [c["case_id"] for c in cases], "annotators": list(annotators),
        "packet": {"path": "packet.json", "sha256": state.sha256_file(AUDIT_DIR / "packet.json")}}
    state.atomic_write_json(AUDIT_DIR / "manifest.json", manifest)
    state.atomic_write_json(AUDIT_DIR / "assigned_key.DO_NOT_OPEN.json", key)
    return manifest


AUDIT_ANNOTATORS = ("annotator1", "annotator2", "annotator3")   # in tie-break order
CONSENSUS_RULE = {   # the earlier audit's consensus rule, reused unchanged
    "risk": "label chosen by at least two of three annotators",
    "factors": "factor kept if ticked by at least two of three",
    "evidence": "phrases of the majority-risk annotator whose evidence has the highest mean phrase F1 "
                "with the other two; none for Indicator",
    "unclear_flags": "set if at least two of three set them",
    "confidence": "minimum among annotators in the risk majority",
    "naturalness": "median of the three", "issue_tags": "tags set by at least two of three"}


def _audit_tool(tool_dir: Path = Path("audit")):
    import importlib.util
    spec = importlib.util.spec_from_file_location("annotate", tool_dir / "annotate.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def audit_consensus() -> dict:
    """2-of-3 consensus from the three complete annotation files. Reads no key."""
    tool = _audit_tool()
    manifest = json.loads((AUDIT_DIR / "manifest.json").read_text())
    ann = {n: json.loads((AUDIT_DIR / "annotations" / f"{n}.json").read_text()) for n in AUDIT_ANNOTATORS}
    for n, a in ann.items():
        assert a["status"] == "complete" and a["packet_sha256"] == manifest["packet"]["sha256"], n
    out = {}
    for cid in manifest["case_ids"]:
        r = {n: ann[n]["responses"][cid] for n in AUDIT_ANNOTATORS}
        votes = Counter(x["risk"] for x in r.values())
        risk, top = votes.most_common(1)[0]
        risk = risk if top >= 2 else "Unclear"
        major = [n for n in AUDIT_ANNOTATORS if r[n]["risk"] == risk] or list(AUDIT_ANNOTATORS)
        if risk == "Indicator":
            evidence, ev_from = [], None
        else:
            def agree(n):
                return statistics.mean(tool.phrase_f1_post(r[n]["evidence"], r[m]["evidence"])
                                       for m in AUDIT_ANNOTATORS if m != n)
            ev_from = max(major, key=lambda n: (agree(n), -AUDIT_ANNOTATORS.index(n)))
            evidence = list(r[ev_from]["evidence"])
        fac = Counter(f for x in r.values() for f in set(x["factors"]))
        tags = Counter(t for x in r.values() for t in set(x["issue_tags"]))
        out[cid] = {
            "risk": risk, "evidence": evidence,
            "evidence_unclear": sum(x["evidence_unclear"] for x in r.values()) >= 2,
            "factors": [f for f in PC.FACTORS_24 if fac[f] >= 2],
            "factors_unclear": sum(x["factors_unclear"] for x in r.values()) >= 2,
            "confidence": min(r[n]["confidence"] for n in major),
            "naturalness": statistics.median(x["naturalness"] for x in r.values()),
            "issue_tags": sorted(t for t, c in tags.items() if c >= 2),
            "notes": f"Majority rule. Risk votes: {dict(votes)}; evidence from {ev_from}",
            "source": "agreement" if top == 3 else "majority"}
    cons = {"protocol": manifest["protocol"], "annotator": "consensus",
            "adjudicator": "2-of-3 majority rule",
            "rule": CONSENSUS_RULE, "source_annotators": list(AUDIT_ANNOTATORS),
            "packet_sha256": manifest["packet"]["sha256"], "status": "complete", "responses": out}
    state.atomic_write_json(AUDIT_DIR / "adjudication" / "consensus.json", cons)
    return cons


def _fleiss(items: list[list[str]], cats: list[str]) -> float:
    import numpy as np
    X = np.array([[sum(lab == c for lab in labs) for c in cats] for labs in items], float)
    k = X.sum(1)[0]
    p = X.sum(0) / X.sum()
    pe = (p ** 2).sum()
    return float((((X * (X - 1)).sum(1) / (k * (k - 1))).mean() - pe) / (1 - pe))


def audit_report() -> dict:
    """Agreement among annotators, then (unblinded) consensus and each annotator
    against the programmed labels, overall and per writer (claim C1)."""
    tool = _audit_tool()
    manifest = json.loads((AUDIT_DIR / "manifest.json").read_text())
    ids = manifest["case_ids"]
    ann = {n: json.loads((AUDIT_DIR / "annotations" / f"{n}.json").read_text()) for n in AUDIT_ANNOTATORS}
    cons = json.loads((AUDIT_DIR / "adjudication" / "consensus.json").read_text())
    agreement = {
        "risk_all_three_agree": sum(len({ann[n]["responses"][i]["risk"] for n in AUDIT_ANNOTATORS}) == 1 for i in ids),
        "risk_fleiss_kappa": _fleiss([[ann[n]["responses"][i]["risk"] for n in AUDIT_ANNOTATORS] for i in ids],
                                     list(PC.RISK_CLASSES) + ["Unclear"]),
        "factor_mean_fleiss_kappa": statistics.mean(
            k for f in PC.FACTORS_24
            if not math.isnan(k := _fleiss([[str(f in ann[n]["responses"][i]["factors"]) for n in AUDIT_ANNOTATORS]
                                            for i in ids], ["True", "False"]))),
        "pairwise": {f"{a}/{b}": tool._pairwise_report(ann[a], ann[b], ids)
                     for a, b in __import__("itertools").combinations(AUDIT_ANNOTATORS, 2)},
        "majority_cases": {i: cons["responses"][i]["notes"] for i in ids if cons["responses"][i]["source"] != "agreement"},
    }
    # ---- unblinding: only after all three annotations and the consensus exist
    raw = json.loads((AUDIT_DIR / "assigned_key.DO_NOT_OPEN.json").read_text())
    key = {cid: {"assigned": v} for cid, v in raw.items()}
    groups = {"all": ids, **{arm: [i for i in ids if raw[i]["arm"] == arm] for arm in E2_ARMS}}
    readers = {**ann, "consensus": cons}
    fidelity = {g: {n: tool._fidelity_report(a, key, gi) for n, a in readers.items()} for g, gi in groups.items()}
    target = {g: {f: [sum(f in cons["responses"][i]["factors"] for i in gi if raw[i]["target_factor"] == f),
                      sum(raw[i]["target_factor"] == f for i in gi)] for f in SC.RARE_FACTORS}
              for g, gi in groups.items() if g != "all"}
    mismatches = {g: dict(Counter(f"{raw[i]['risk']}->{cons['responses'][i]['risk']}" for i in gi
                                  if raw[i]["risk"] != cons["responses"][i]["risk"]))
                  for g, gi in groups.items() if g != "all"}
    by_case = {}
    for i in ids:
        by_case.setdefault(raw[i]["source_case_id"], {})[raw[i]["arm"]] = cons["responses"][i]["risk"] == raw[i]["risk"]
    paired = Counter((v["pkgc_sonnet"], v["pkgc_gemma"]) for v in by_case.values())
    rep = {"agreement": agreement, "fidelity": fidelity, "target_factor_affirmed_consensus": target,
           "risk_mismatches_consensus": mismatches,
           "paired_risk_match": {"both": paired[(True, True)], "sonnet_only": paired[(True, False)],
                                 "gemma_only": paired[(False, True)], "neither": paired[(False, False)]},
           "naturalness_mean_by_writer": {arm: statistics.mean(ann[n]["responses"][i]["naturalness"]
                                                              for n in AUDIT_ANNOTATORS for i in groups[arm])
                                          for arm in E2_ARMS}}
    state.atomic_write_json(AUDIT_DIR / "report.json", rep)
    return rep
