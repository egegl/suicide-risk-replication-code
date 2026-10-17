"""Freeze every number the ICHI 2027 manuscript, its supplement, and its figures report (E2).

Run from the repo root (needs pandas + pyarrow for the processed data):

    PYTHONDONTWRITEBYTECODE=1 python paper/figures/freeze.py [--rebuilt]

Reads the saved grouped out-of-fold predictions of the five training conditions, the generated
training files, the writer and labeler outputs, and the human audit, and writes frozen.json next
to this file. Nothing is retrained and no API is called. Before writing, it reproduces the
registered E2 readout (runs31-pkgc/reanalysis/e2_confirm.json, from scripts/e2_confirm.py) and
the paper's control composites, and stops on any mismatch. --rebuilt (for retrained runs, which
are not bit-identical) reports mismatches instead of stopping. Output holds aggregates only: no
post text, quotes, user ids, or row ids (data use agreement).

scripts/e2_confirm.py imports Scorer, draws, replicate_table, summarize, use, read_jsonl, preds,
expect, clean, METRICS, P, and INPUTS from this file; keep their behaviour unchanged.
"""
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import scorer  # noqa: E402
from prep.constants import FACTOR_INDEX, FACTORS_24, RISK_CLASSES  # noqa: E402
from synth.constants import RARE_FACTORS  # noqa: E402
from train import ensemble, oof, reanalysis, thresholds  # noqa: E402
from train.constants import BOOTSTRAP_B, BOOTSTRAP_SEED, CHECKPOINT_STEPS, GRAD_ACCUM  # noqa: E402

OUT = Path(__file__).resolve().parent / "frozen.json"
P = ROOT / "data" / "processed"
ARMS = {  # paper name -> (runs dir, arm id)
    "real_only": ("runs31", "baseline"),
    "repeated_real": ("runs31", "qcontrol"),
    "rare_weighted": ("runs31-rare", "rareweight"),  # real rare-factor posts weighted 3x in the loss
    "sonnet": ("runs31-pkgc", "pkgc_sonnet"),
    "gemma": ("runs31-pkgc", "pkgc_gemma"),
}
WRITERS = {  # paper name -> (E2 arm id, assembled files, writer output name)
    "sonnet": ("pkgc_sonnet", "data/synthetic_pkgc/arms/sonnet_pkgc-v2", "sonnet"),
    "gemma": ("pkgc_gemma", "data/synthetic_pkgc/arms/gemma31_pkgc-v2", "gemma31"),
}
CONTRASTS = {
    "sonnet_vs_real_only": ("sonnet", "real_only"),          # P1 (registered primary)
    "gemma_vs_real_only": ("gemma", "real_only"),            # P2 (registered primary)
    "repeated_real_vs_real_only": ("repeated_real", "real_only"),
    "rare_weighted_vs_real_only": ("rare_weighted", "real_only"),
    "sonnet_vs_repeated_real": ("sonnet", "repeated_real"),
    "gemma_vs_repeated_real": ("gemma", "repeated_real"),
    "sonnet_vs_rare_weighted": ("sonnet", "rare_weighted"),
    "gemma_vs_rare_weighted": ("gemma", "rare_weighted"),
    "sonnet_vs_gemma": ("sonnet", "gemma"),
}
E2_KEYS = {"sonnet_vs_real_only": ("primary", "P1"), "gemma_vs_real_only": ("primary", "P2"),
           "sonnet_vs_repeated_real": ("secondary", "sonnet_vs_repeated_real"),
           "gemma_vs_repeated_real": ("secondary", "gemma_vs_repeated_real"),
           "sonnet_vs_rare_weighted": ("secondary", "sonnet_vs_rare_weighted"),
           "gemma_vs_rare_weighted": ("secondary", "gemma_vs_rare_weighted"),
           "sonnet_vs_gemma": ("secondary", "sonnet_vs_gemma")}
SEEDS = (1, 2)
RARE = [FACTOR_INDEX[f] for f in RARE_FACTORS]
OTHER = [i for i in range(len(FACTORS_24)) if i not in RARE]
METRICS = ("composite", "risk_wf1", "phrase_f1", "factor_macro_f1", "rare6_macro_f1", "other18_macro_f1")
NO_EVIDENCE = "composite_no_evidence"  # registered robustness reading: (0.4 risk + 0.3 factors) / 0.7
INPUTS = {}
STRICT = "--rebuilt" not in sys.argv


def use(rel):
    path = ROOT / rel
    INPUTS[str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_json(rel):
    return json.loads(use(rel).read_text())


def read_jsonl(rel):
    return [json.loads(x) for x in use(rel).read_text().splitlines() if x.strip()]


def expect(label, got, want, tol=1e-9):
    ok = abs(got - want) <= tol if isinstance(want, float) else got == want
    if not ok:
        if STRICT:
            raise SystemExit(f"MISMATCH {label}: got {got!r}, want {want!r}")
        print(f"differs from the paper: {label}: got {got!r}, paper {want!r}")


def preds(rows):
    return {r["row_id"]: {"risk": r["risk"], "spans": list(r["spans"]), "factors": list(r["factors"])}
            for r in rows}


# ------------------------------------------------------------------ scoring on row weights
class Scorer:
    """Composite and components for any row-multiplicity vector m (m = 1 is the full data).

    Same definitions as src/scorer.py (risk weighted F1 over the four levels, per-post phrase
    F1 averaged with empty-vs-empty = 1, factor macro F1 over 24 labels, zero_division = 0),
    written as counts so 2,000 user-bootstrap replicates run in seconds. Checked against
    scorer.score on the full data and against the registered bootstrap deltas."""

    def __init__(self, gold, ids):
        self.ids = ids
        self.risk_true = np.array([RISK_CLASSES.index(gold[i]["risk"]) for i in ids])
        self.G = np.zeros((len(ids), 24), dtype=np.float64)
        for r, i in enumerate(ids):
            for f in gold[i]["factors"]:
                self.G[r, FACTOR_INDEX[f]] = 1
        self.gold = gold

    def label(self, full):
        ids = self.ids
        risk = np.array([RISK_CLASSES.index(full[i]["risk"]) for i in ids])
        phrase = np.array([scorer.phrase_f1_post(full[i]["spans"], self.gold[i]["spans"]) for i in ids])
        Pm = np.zeros((len(ids), 24), dtype=np.float64)
        for r, i in enumerate(ids):
            for f in full[i]["factors"]:
                Pm[r, FACTOR_INDEX[f]] = 1
        return {"risk_code": self.risk_true * 4 + risk, "phrase": phrase,
                "TP": self.G * Pm, "FP": (1 - self.G) * Pm, "FN": self.G * (1 - Pm)}

    @staticmethod
    def f1(tp, fp, fn):
        den = 2 * tp + fp + fn
        return np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)

    def score(self, lab, m):
        cm = np.bincount(lab["risk_code"], weights=m, minlength=16).reshape(4, 4)
        tp = np.diag(cm)
        support = cm.sum(1)
        f1_risk = self.f1(tp, cm.sum(0) - tp, support - tp)
        wf1 = float((support * f1_risk).sum() / support.sum()) if support.sum() else 0.0
        phrase = float(m @ lab["phrase"] / m.sum())
        per_factor = self.f1(m @ lab["TP"], m @ lab["FP"], m @ lab["FN"])
        macro = float(per_factor.mean())
        return {"composite": 0.4 * wf1 + 0.3 * phrase + 0.3 * macro, "risk_wf1": wf1, "phrase_f1": phrase,
                "factor_macro_f1": macro, "rare6_macro_f1": float(per_factor[RARE].mean()),
                "other18_macro_f1": float(per_factor[OTHER].mean()), "per_factor_f1": per_factor}


def draws(gold, users, ids):
    """The registered shared draws (src/train/stats.py paired_bootstrap)."""
    uniq = sorted({users[i] for i in ids})
    rows_by_user = {u: [] for u in uniq}
    for r, i in enumerate(ids):
        rows_by_user[users[i]].append(r)
    user_rows = [np.array(rows_by_user[u], dtype=np.intp) for u in uniq]
    d = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(uniq), size=(BOOTSTRAP_B, len(uniq)))
    weights = np.empty((BOOTSTRAP_B, len(ids)))
    for b in range(BOOTSTRAP_B):
        weights[b] = np.bincount(np.concatenate([user_rows[j] for j in d[b]]), minlength=len(ids))
    return weights, len(uniq)


def replicate_table(sc, labs, W):
    """label -> metric -> array of replicate values."""
    out = {}
    for name, lab in labs.items():
        reps = {k: np.empty(len(W)) for k in METRICS}
        for b, m in enumerate(W):
            s = sc.score(lab, m)
            for k in METRICS:
                reps[k][b] = s[k]
        out[name] = reps
    return out


def summarize(d):
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"mean": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "positive_fraction": float(np.mean(d > 0))}


def no_evidence(x):
    """The evidence-free reading: risk and factor parts of the composite, rescaled to sum to 1."""
    return (0.4 * x["risk_wf1"] + 0.3 * x["factor_macro_f1"]) / 0.7


def two_sided_p(d):
    return float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))


# ------------------------------------------------------------------ blocks
def dataset(gold, posts):
    tr = posts[posts.split == "train"]
    support = Counter(f for g in gold.values() for f in set(g["factors"]))
    out = {
        "n_posts": len(tr), "n_users": int(tr.anon_user_id.nunique()),
        "risk_counts": {c: sum(g["risk"] == c for g in gold.values()) for c in RISK_CLASSES},
        "factor_support": {f: support[f] for f in FACTORS_24},
        "factor_order": list(FACTORS_24),  # 19 risk factors, then 5 protective (frozen.json sorts keys)
        "rare6": list(RARE_FACTORS),
        "n_no_factor": sum(not g["factors"] for g in gold.values()),
        "folds_users": tr.groupby("fold").anon_user_id.nunique().tolist(),
        "folds_posts": tr.groupby("fold").size().tolist(),
        "indicator_posts_with_evidence": sum(g["risk"] == "Indicator" and any(scorer.norm(x) for x in g["spans"])
                                             for g in gold.values()),
        "factors_per_post": float(np.mean([len(g["factors"]) for g in gold.values()])),
        "factors_per_post_with_rare_factor": float(np.mean(
            [len(g["factors"]) for g in gold.values() if set(RARE_FACTORS) & set(g["factors"])])),
    }
    expect("n_posts", out["n_posts"], 1635)
    expect("n_users", out["n_users"], 153)
    expect("risk counts", out["risk_counts"], {"Indicator": 611, "Ideation": 519, "Behavior": 391, "Attempt": 114})
    expect("rare6 supports", [support[f] for f in RARE_FACTORS], [33, 33, 16, 14, 8, 45])
    expect("support range", (min(support.values()), max(support.values())), (8, 745))
    expect("no-factor posts", out["n_no_factor"], 128)
    expect("folds", (out["folds_users"], out["folds_posts"]), ([29, 31, 34, 30, 29], [331, 326, 332, 325, 321]))
    return out


def focal(row):
    """The focal post of a training row (the text between the focal marker and any later post)."""
    return row["input"].split("[FOCAL POST]\n", 1)[1].split("\n[Later post", 1)[0]


def training_files():
    """Rows per training file, generated-row composition, and how often each real row is seen."""
    out = {"writers": {}}
    q = read_json("data/synthetic/arms/qcontrol/manifest.json")["folds"]
    for name, (arm, d, _) in WRITERS.items():
        m = read_json(f"{d}/manifest.json")["folds"]
        rows = {"real": [], "generated": []}
        multiple = []  # generated over real positive rows, per target factor and training file
        for k in range(5):
            real_pos = Counter()
            for line in (P / "sft" / f"train_fold{k}.jsonl").read_text().splitlines():
                real_pos.update(f for f, v in json.loads(json.loads(line)["target"])["factors"].items() if v)
            multiple += [m[str(k)]["rare_positives"][f] / real_pos[f] for f in RARE_FACTORS]
            for r in read_jsonl(f"{d}/train_fold{k}.jsonl"):
                t = json.loads(r["target"])
                ev = [x for x in t["evidence"] if scorer.norm(x)]
                rec = (t["risk"], ev, [f for f, v in t["factors"].items() if v], len(focal(r).split()))
                rows["generated" if "synth" in r else "real"].append(rec)
        expect(f"{name} sizes = repeated real", [m[str(k)]["total"] for k in range(5)],
               [q[str(k)]["n_real"] for k in range(5)])

        def describe(rs):
            ev = [s for risk, s, _, _ in rs if risk != "Indicator"]
            lens = [scorer.toklen(x) for s in ev for x in s]
            return {"n": len(rs), "risk_share": {c: sum(r[0] == c for r in rs) / len(rs) for c in RISK_CLASSES},
                    "factors_per_row": float(np.mean([len(f) for _, _, f, _ in rs])),
                    "spans_per_evidence_row": float(np.mean([len(s) for s in ev])),
                    "tokens_per_span_mean": float(np.mean(lens)),
                    "words_mean": float(np.mean([w for *_, w in rs]))}

        out["writers"][name] = {
            "real_rows_per_fold": [m[str(k)]["real"] for k in range(5)],
            "generated_rows_per_fold": [m[str(k)]["synthetic"] for k in range(5)],
            "rare_positives": {f: sum(m[str(k)]["rare_positives"][f] for k in range(5)) for f in RARE_FACTORS},
            "positive_multiple_range": [min(multiple), max(multiple)],
            "real_rows": describe(rows["real"]), "generated_rows": describe(rows["generated"])}
    a, b = (out["writers"][w] for w in WRITERS)
    expect("both writers fill the same dose", a["generated_rows_per_fold"], b["generated_rows_per_fold"])
    expect("dose", a["generated_rows_per_fold"], [1211, 1309, 1309, 1234, 1321])
    real, gen = a["real_rows_per_fold"], a["generated_rows_per_fold"]
    seen = CHECKPOINT_STEPS[-1] * GRAD_ACCUM  # effective batch 16, micro-batch 1
    out.update(real_rows_per_fold=real, generated_rows_per_fold=gen,
               generated_share_mean=float(np.mean([g / (r + g) for r, g in zip(real, gen)])),
               passes_over_real_at_160={"real_only": float(np.mean([seen / r for r in real])),
                                        "augmented": float(np.mean([seen / (r + g) for r, g in zip(real, gen)]))})
    return out


def fidelity(rows):
    """rows: (programmed risk, programmed factors, target factor, labeler risk, labeler factors)."""
    tp = fp = fn = 0
    hit = defaultdict(list)
    for risk, fac, target, lrisk, lfac in rows:
        prog, pred = set(fac), set(lfac)
        tp += len(prog & pred); fp += len(pred - prog); fn += len(prog - pred)
        hit[target].append(target in pred)
    p, r = tp / (tp + fp), tp / (tp + fn)
    by = {f: float(np.mean(hit[f])) for f in RARE_FACTORS}
    lvl = RISK_CLASSES.index
    return {"n": len(rows), "risk_agreement": float(np.mean([a == b for a, _, _, b, _ in rows])),
            "risk_read_higher": float(np.mean([lvl(b) > lvl(a) for a, _, _, b, _ in rows])),
            "risk_read_lower": float(np.mean([lvl(b) < lvl(a) for a, _, _, b, _ in rows])),
            "risk_confusion": dict(Counter(f"{a}->{b}" for a, _, _, b, _ in rows if a != b)),
            "target_recall_pooled": float(np.mean([t for v in hit.values() for t in v])),
            "target_recall_by_factor": by, "target_recall_balanced": float(np.mean(list(by.values()))),
            "n_by_factor": {f: len(hit[f]) for f in RARE_FACTORS},
            "factor_micro": {"precision": p, "recall": r, "f1": 2 * p * r / (p + r)}}


def label_fidelity(gold):
    """Claim C1, automatic half: each fold's real-only model reads every generated row it would be
    tested against (the trained-on rows), and its real held-out posts that carry a rare factor."""
    out = {}
    for name, (arm, d, wname) in WRITERS.items():
        rows = []
        for k in range(5):
            labels = {l["case_id"]: l for l in read_jsonl(f"data/synthetic_pkgc/full/pkgc-v2/fold{k}/labels/{wname}.jsonl")}
            for r in read_jsonl(f"{d}/train_fold{k}.jsonl"):
                if "synth" not in r:
                    continue
                t, lab = json.loads(r["target"]), labels[r["synth"]["case_id"]]
                assert not lab.get("parse_failure")
                rows.append((t["risk"], [f for f, v in t["factors"].items() if v], r["synth"]["target_factor"],
                             lab["risk"], lab["factors"]))
        out[name] = fidelity(rows)
    real = []
    for k in range(5):
        for r in read_jsonl(f"runs31/baseline-f{k}-s1/oof.jsonl"):
            g = gold[r["row_id"]]
            real += [(g["risk"], g["factors"], f, r["risk"], r["factors"])
                     for f in RARE_FACTORS if f in g["factors"]]
    out["real_held_out"] = fidelity(real)
    expect("real labeler baseline n", out["real_held_out"]["n"], 142)
    expect("real labeler baseline balanced recall", round(out["real_held_out"]["target_recall_balanced"], 2), 0.58)
    return out


def generation():
    """Cases written, check outcomes, and Sonnet's token use (case counts from each fold's files)."""
    out = {}
    for name, (arm, d, wname) in WRITERS.items():
        m = read_json(f"{d}/manifest.json")["folds"]
        cases, status, words, usage = 0, Counter(), [], Counter()
        for k in range(5):
            outs = read_jsonl(f"data/synthetic_pkgc/full/pkgc-v2/fold{k}/outputs/{wname}.jsonl")
            cases += len(read_jsonl(f"data/synthetic_pkgc/full/pkgc-v2/fold{k}/cases.jsonl"))
            status.update(o["status"] for o in outs)
            for o in outs:
                usage.update({x: v for x, v in (o.get("usage") or {}).items() if isinstance(v, int)})
        rejects = Counter()
        for k in range(5):
            rejects.update({x: v for x, v in m[str(k)]["rejects_before_dose"].items() if x != "no_evidence_span"})
        out[name] = {"cases": cases, "status": dict(status), "check_rejects": dict(rejects),
                     "passed_checks": cases - sum(rejects.values()),
                     "no_evidence_span_before_dose": sum(m[str(k)]["rejects_before_dose"].get("no_evidence_span", 0)
                                                         for k in range(5)),
                     "usage": dict(usage)}
    u = out["sonnet"]["usage"]
    out["sonnet"]["cost_usd_batch"] = (u["input_tokens"] * 1.0 + u["output_tokens"] * 5.0) / 1e6  # $1 / $5 per MTok
    return out


def audit():
    """Claim C1, human half: the registered audit (data/synthetic_pkgc/audit/e2_v1)."""
    d = "data/synthetic_pkgc/audit/e2_v1/"
    rep = read_json(f"{d}report.json")
    man = read_json(f"{d}manifest.json")
    read_json(f"{d}adjudication/consensus.json")
    ag = rep["agreement"]
    pw = ag["pairwise"].values()
    out = {"n_posts": len(man["case_ids"]), "per_writer": 24, "per_factor": man["per_factor"], "seed": man["seed"],
           "annotators": len(man["annotators"]),
           "risk_all_three_agree": ag["risk_all_three_agree"], "risk_fleiss_kappa": ag["risk_fleiss_kappa"],
           "factor_mean_fleiss_kappa": ag["factor_mean_fleiss_kappa"],
           "pairwise_risk_exact": [min(p["risk_exact_agreement"] for p in pw), max(p["risk_exact_agreement"] for p in pw)],
           "pairwise_factor_micro_f1": [min(p["factor_micro"]["f1"] for p in pw), max(p["factor_micro"]["f1"] for p in pw)],
           "pairwise_evidence_f1": [min(p["evidence_mean_bidirectional_phrase_f1"] for p in pw),
                                    max(p["evidence_mean_bidirectional_phrase_f1"] for p in pw)],
           "majority_decided": len(ag["majority_cases"]),
           "paired_risk_match": rep["paired_risk_match"], "writers": {}}
    for name, (arm, _, _) in WRITERS.items():
        c = rep["fidelity"][arm]["consensus"]
        out["writers"][name] = {
            "risk_matches": round(c["risk_exact_accuracy_vs_assigned"] * c["n_cases"]),
            "factor_micro": {k: c["factor_micro_vs_assigned"][k] for k in ("precision", "recall", "f1")},
            "exact_factor_set": c["factor_exact_set_rate_vs_assigned"],
            "evidence_phrase_f1": c["evidence_mean_phrase_f1_vs_assigned"],
            "naturalness_mean": rep["naturalness_mean_by_writer"][arm],
            "target_affirmed": rep["target_factor_affirmed_consensus"][arm],
            "risk_mismatches": rep["risk_mismatches_consensus"][arm]}
    expect("audit design", (out["n_posts"], out["per_factor"]), (48, 4))
    expect("audit kappa", round(out["risk_fleiss_kappa"], 3), 0.843)
    return out


def pilot_survey():
    """The fold-0 writer pilots (300 cases, exploratory) behind the prompt and writer choices."""
    out = {}
    for version, path in (("v1", "data/synthetic_pkgc/pilot/report.json"), ("v2", "data/synthetic_pkgc/pilot_v2/report.json")):
        rep = read_json(path)
        out[version] = {}
        for arm, a in rep["arms"].items():
            lf = a["label_fidelity"]
            out[version][arm] = {
                "cases": a["n_cases"], "passed_checks": a["accepted"], "rejects": a["rejects"],
                "risk_agreement": lf["risk_agreement"], "target_recall": lf["target_recall"],
                "factor_micro_f1": lf["factor_micro_f1"], "taxonomy_vocabulary": a["taxonomy_vocabulary"],
                "words_mean": a["style"]["mean_words"], "realism_auc": a["realism_leakfree"]["auc_mean"]}
        out[version + "_real_reference"] = {"taxonomy_vocabulary": rep["real_reference"]["taxonomy_vocabulary"],
                                            "words_mean": rep["real_reference"]["style"]["mean_words"]}
    return out


def prompts():
    """The writer prompts as sent (no post text): system prompts, the user-prompt layout, and the
    factor definitions where the prompt departs from the task's."""
    from synth import pkgc
    case = {"earlier_ids": ["e1", "e2"], "later_ids": ["l1"], "risk": "<risk level>",
            "factors": ["hopelessness"], "approx_words": "<n>"}
    texts = {"e1": "<earlier post>", "e2": "<earlier post>", "l1": "<later post>"}
    user = pkgc.render_user_prompt(case, texts).replace(
        f"- hopelessness: {pkgc.definition('hopelessness')}", "- <factor>: <definition> (one line per factor)")
    assert "<definition>" in user
    return {"system_v1": pkgc.SYSTEM_PROMPTS["pkgc-v1"], "system_v2": pkgc.SYSTEM_PROMPTS["pkgc-v2"],
            "user_layout": user, "overrides_v2": pkgc.DEFINITION_OVERRIDES["pkgc-v2"],
            "reference_chars": pkgc.REFERENCE_CHARS, "oversample": pkgc.FULL_OVERSAMPLE}


def submitted_system(gold):
    """The competition entry (two seeds merged, bagged cutoffs): confusion matrix and per-factor profile."""
    merged = ensemble.merge_rows(read_jsonl("runs31/pooled/baseline-s1.oof.jsonl"),
                                 read_jsonl("runs31/pooled/baseline-s2.oof.jsonl"))
    rows, _ = thresholds.nested_thresholded_pool(merged, gold, estimator="bagged")
    full = oof.propagate(preds(rows), P)
    cm = np.zeros((4, 4), dtype=int)
    for i, g in gold.items():
        cm[RISK_CLASSES.index(g["risk"]), RISK_CLASSES.index(full[i]["risk"])] += 1
    expect("submitted confusion matrix", cm.tolist(),
           [[530, 62, 15, 4], [34, 419, 62, 4], [14, 45, 319, 13], [3, 4, 19, 88]])
    profile = {}
    for f in FACTORS_24:
        tp = sum(f in g["factors"] and f in full[i]["factors"] for i, g in gold.items())
        n_pred = sum(f in full[i]["factors"] for i in gold)
        n_gold = sum(f in g["factors"] for g in gold.values())
        prec, rec = tp / n_pred, tp / n_gold
        profile[f] = {"support": n_gold, "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec)}
    return {"confusion_matrix": cm.tolist(), "order": list(RISK_CLASSES), "factor_profile": profile}


def factor_behaviour(gold, ids, fulls):
    G = np.array([[f in gold[i]["factors"] for f in FACTORS_24] for i in ids])
    out = {"gold_factors_per_post": float(G.sum(1).mean())}
    lvl = RISK_CLASSES.index
    out["gold_risk_share"] = {c: float(np.mean([gold[i]["risk"] == c for i in ids])) for c in RISK_CLASSES}
    for arm, per_seed in fulls.items():
        dens, prec, rec, above, below, share = [], [], [], [], [], []
        for full in per_seed:
            Pm = np.array([[f in full[i]["factors"] for f in FACTORS_24] for i in ids])
            tp = (Pm & G).sum()
            dens.append(Pm.sum(1).mean()); prec.append(tp / Pm.sum()); rec.append(tp / G.sum())
            above.append(sum(lvl(full[i]["risk"]) > lvl(gold[i]["risk"]) for i in ids))
            below.append(sum(lvl(full[i]["risk"]) < lvl(gold[i]["risk"]) for i in ids))
            share.append([np.mean([full[i]["risk"] == c for i in ids]) for c in RISK_CLASSES])
        out[arm] = {"factors_per_post": float(np.mean(dens)), "micro_precision": float(np.mean(prec)),
                    "micro_recall": float(np.mean(rec)), "risk_calls_above_annotated": float(np.mean(above)),
                    "risk_calls_below_annotated": float(np.mean(below)),
                    "predicted_risk_share": dict(zip(RISK_CLASSES, np.mean(share, 0).tolist()))}
    return out


def step_rows(runs, arm, seed, step, gold):
    rows = []
    for k in range(5):
        run_id = f"{arm}-f{k}-s{seed}"
        kind, path = reanalysis.resolve_step_source(ROOT / runs, run_id, step)
        use(str(path.relative_to(ROOT)))
        rows += reanalysis.fold_rows_at_step(ROOT / runs, run_id, step)
    rows.sort(key=lambda r: (r["fold"], r["row_id"]))
    thr, _ = thresholds.nested_thresholded_pool(rows, gold)
    return thr


def main():
    os.chdir(ROOT)
    gold = scorer.load_gold(use("data/processed/gold.jsonl"))
    posts = pd.read_parquet(use("data/processed/posts.parquet"), columns=["row_id", "split", "anon_user_id", "fold"])
    users = dict(zip(posts.row_id, posts.anon_user_id))
    ids = sorted(gold)
    sc = Scorer(gold, ids)
    e2 = read_json("runs31-pkgc/reanalysis/e2_confirm.json")
    W, n_users = draws(gold, users, ids)

    def condition_block(variant):
        """Scores and labels of every condition under a checkpoint variant ('' = as registered)."""
        labs, fulls, per_seed = {}, {}, {}
        suffix = f".{variant}" if variant else ""
        for name, (runs, arm) in ARMS.items():
            path = f"{runs}/pooled/{arm}-s1{suffix}.thresholded.jsonl"
            if not (ROOT / path).exists():
                continue  # rare weighting has no nested re-pool (its per-step predictions were not kept)
            for s in SEEDS:
                full = oof.propagate(preds(read_jsonl(f"{runs}/pooled/{arm}-s{s}{suffix}.thresholded.jsonl")), P)
                fulls.setdefault(name, []).append(full)
                labs[f"{name}-s{s}"] = sc.label(full)
                got = sc.score(labs[f"{name}-s{s}"], np.ones(len(ids)))
                if not variant:
                    ref = scorer.score(full, gold)
                    expect(f"{name} s{s} composite vs scorer.score", got["composite"], ref["composite"], 1e-12)
                    expect(f"{name} s{s} macro vs scorer.score", got["factor_macro_f1"], ref["macro_f1"], 1e-12)
                    key = WRITERS[name][0] if name in WRITERS else name
                    expect(f"{name} s{s} composite vs e2_confirm", round(got["composite"], 8),
                           e2["per_seed"][key][str(s)]["composite"], 1e-8)
                per_seed.setdefault(name, {})[str(s)] = {k: got[k] for k in METRICS}
                per_seed[name][str(s)]["per_factor_f1"] = dict(zip(FACTORS_24, got["per_factor_f1"].tolist()))
        conditions = {}
        for name, a in per_seed.items():
            mean = {k: float(np.mean([a[s][k] for s in a])) for k in METRICS}
            mean[NO_EVIDENCE] = no_evidence(mean)
            conditions[name] = {"per_seed": a, "mean_seed": mean,
                                "per_factor_f1_mean_seed": {f: float(np.mean([a[s]["per_factor_f1"][f] for s in a]))
                                                            for f in FACTORS_24}}
        reps = replicate_table(sc, labs, W)
        arm_reps = {}
        for name in conditions:
            arm_reps[name] = {k: (reps[f"{name}-s1"][k] + reps[f"{name}-s2"][k]) / 2 for k in METRICS}
            arm_reps[name][NO_EVIDENCE] = no_evidence(arm_reps[name])
        paired = {}
        for key, (t, r) in CONTRASTS.items():
            if t not in conditions or r not in conditions:
                continue
            paired[key] = {}
            for k in (*METRICS, NO_EVIDENCE):
                d = arm_reps[t][k] - arm_reps[r][k]
                paired[key][k] = {**summarize(d), "p_two_sided": two_sided_p(d),
                                  "observed": conditions[t]["mean_seed"][k] - conditions[r]["mean_seed"][k]}
        return conditions, paired, fulls

    # ---- the five conditions at the registered checkpoints, and the registered readout
    conditions, paired, fulls = condition_block("")
    for key, (block, e2key) in E2_KEYS.items():
        for k in METRICS:
            for stat in ("mean", "ci_low", "ci_high", "p_two_sided"):
                expect(f"{key} {k} {stat} vs e2_confirm", round(paired[key][k][stat], 8), e2[block][e2key][k][stat], 1e-8)
    expect("controls reproduce the registered composites",
           [round(conditions[n]["mean_seed"]["composite"], 4) for n in ("real_only", "repeated_real", "rare_weighted")],
           [0.7751, 0.7734, 0.7729])
    registered = {"holm": e2["holm"], "adoption": {w: e2["adoption"][WRITERS[w][0]] for w in WRITERS},
                  "ship_bar": e2["adoption"]["pkgc_sonnet"]["ship_bar"]}

    # ---- registered robustness: final checkpoint and checkpoint chosen on the other folds
    robustness = {}
    for variant in ("fixed160", "nested"):
        cond_v, paired_v, _ = condition_block(variant)
        robustness[variant] = {"mean_seed": {n: c["mean_seed"] for n, c in cond_v.items()}, "paired": paired_v}
    for variant in ("fixed160", "nested"):
        path = f"runs31/reanalysis/{variant}_sweep_table.json"  # the competition sweep's own readout
        if not (ROOT / path).exists():
            continue
        table = read_json(path)
        for name, arm in (("real_only", "baseline"), ("repeated_real", "qcontrol")):
            want = np.mean([table["arms"][arm]["per_seed"][s]["thresholded"]["composite_ee1"] for s in ("1", "2")])
            expect(f"{variant} {name} vs runs31 table", robustness[variant]["mean_seed"][name]["composite"], float(want), 1e-12)

    # ---- composite at every saved checkpoint (supplement)
    steps, cache = {}, {}
    for name in ("real_only", "repeated_real", "sonnet", "gemma"):
        runs, arm = ARMS[name]
        for st in CHECKPOINT_STEPS:
            labs = [sc.label(oof.propagate(preds(step_rows(runs, arm, s, st, gold)), P)) for s in SEEDS]
            vals = [sc.score(l, np.ones(len(ids)))["composite"] for l in labs]
            steps.setdefault(name, {})[str(st)] = {"per_seed": vals, "mean_seed": float(np.mean(vals))}
            cache[(name, st)] = np.array([np.mean([sc.score(l, m)["composite"] for l in labs]) for m in W])
        expect(f"{name} step 160 = fixed160 pool", steps[name]["160"]["mean_seed"],
               robustness["fixed160"]["mean_seed"][name]["composite"], 1e-12)
    by_step = {str(st): {f"{n}_vs_real_only": {**summarize(cache[(n, st)] - cache[("real_only", st)]),
                                               "observed": steps[n][str(st)]["mean_seed"] - steps["real_only"][str(st)]["mean_seed"]}
                         for n in ("repeated_real", "sonnet", "gemma")} for st in CHECKPOINT_STEPS}

    selection = {n: sorted(read_json(f"{ARMS[n][0]}/{ARMS[n][1]}-f{k}-s{s}/selected.json")["step"]
                           for k in range(5) for s in SEEDS) for n in ("real_only", "repeated_real", "sonnet", "gemma")}

    # post hoc: does a factor's F1 change track how much the generated rows raised its share of labels?
    from scipy.stats import spearmanr
    tf = training_files()
    prevalence_vs_f1 = {}
    for w in WRITERS:
        _, d, _ = WRITERS[w]
        real, gen, n = Counter(), Counter(), Counter()
        for k in range(5):
            for r in read_jsonl(f"{d}/train_fold{k}.jsonl"):
                src = "generated" if "synth" in r else "real"
                n[src] += 1
                (gen if src == "generated" else real).update(
                    f for f, v in json.loads(r["target"])["factors"].items() if v)
        ratio = [np.log(((real[f] + gen[f]) / (n["real"] + n["generated"])) / (real[f] / n["real"])) for f in FACTORS_24]
        delta = [conditions[w]["per_factor_f1_mean_seed"][f] - conditions["real_only"]["per_factor_f1_mean_seed"][f]
                 for f in FACTORS_24]
        rho, p = spearmanr(ratio, delta)
        prevalence_vs_f1[w] = {"spearman_rho": float(rho), "p": float(p)}

    frozen = {
        "dataset": {**dataset(gold, posts),
                    "n_distinct_inputs": len(read_jsonl("runs31/pooled/baseline-s1.thresholded.jsonl"))},
        "bootstrap": {"b": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED, "unit": "user", "n_users": n_users,
                      "n_scored_rows": len(ids)},
        "training_files": tf,
        "prevalence_vs_f1": prevalence_vs_f1,
        "conditions": conditions,
        "paired": paired,
        "registered": registered,
        "robustness": robustness,
        "checkpoints": {"curves": steps, "paired_by_step": by_step, "selected_steps": selection},
        "label_fidelity": label_fidelity(gold),
        "audit": audit(),
        "generation": generation(),
        "pilot_survey": pilot_survey(),
        "prompts": prompts(),
        "factor_behaviour": factor_behaviour(gold, ids, fulls),
        "submitted_system": submitted_system(gold),
        "public_leaderboard_real_only_ensemble": {"composite": 0.7644, "subtask1": 0.7991, "subtask2": 0.6834},
        "meaning_in_life_oracle_cutoff_f1": 0.327,  # preliminary real-only runs, best cutoff chosen on the scored data
        "reproduction_gate": {"original": 0.7933, "rerun": 0.7853, "tolerance": 0.010},  # baseline-f0-s1 retrained before E2
        "inputs_sha256": dict(sorted(INPUTS.items())),
    }
    OUT.write_text(json.dumps(clean(frozen), indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return round(float(x), 8) + 0.0
    return x


if __name__ == "__main__":
    main()
