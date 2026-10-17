"""Build the data figures and tables from frozen.json, then check the manuscript's numbers.

    python3 figures.py            # build every figure and table into out/, then check main.tex
    python3 figures.py --check    # build twice and fail unless the outputs are byte-identical

Fig. 1 is a TikZ schematic (fig1_design.tex). Every other mark and table cell is read from frozen.json
(written by freeze.py). Library defaults throughout; each figure is drawn wider than
its printed width so the default text prints small. Point estimates are full-data differences;
intervals are the 2.5th and 97.5th percentiles of the 2,000 paired user-bootstrap replicates.
"""
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
D = json.loads((HERE / "frozen.json").read_text())
MAIN = HERE.parent / "main.tex"
SUPP = HERE.parent / "supplement.tex"
plt.rcParams["svg.hashsalt"] = "ichi2027"
plt.rcParams["pdf.fonttype"] = 42  # TrueType, not Type 3: IEEE PDF eXpress rejects Type 3 fonts

CONTRASTS = [("sonnet_vs_real_only", "Sonnet data vs real only"),
             ("gemma_vs_real_only", "Gemma data vs real only"),
             ("repeated_real_vs_real_only", "Repeated real vs real only"),
             ("rare_weighted_vs_real_only", "Rare-factor weighting vs real only"),
             ("sonnet_vs_repeated_real", "Sonnet data vs repeated real"),
             ("gemma_vs_repeated_real", "Gemma data vs repeated real"),
             ("sonnet_vs_gemma", "Sonnet data vs Gemma data")]
OUTPUTS = [("composite", "Composite"), ("risk_wf1", "Risk level"), ("phrase_f1", "Evidence"),
           ("factor_macro_f1", "All 24 factors"), ("rare6_macro_f1", "Six target factors"),
           ("other18_macro_f1", "Other 18 factors")]
CONDITIONS = [("Real only", "real_only"), ("Repeated real", "repeated_real"),
              ("Rare-factor weighting", "rare_weighted"), ("Real + Sonnet data", "sonnet"),
              ("Real + Gemma data", "gemma")]
NAMES = {"suicide means (with access)": "Suicide means with access",
         "prior self-harm or suicidal thought/attempt": "Prior self-harm, suicidal thought, or attempt",
         "physical health/characteristic": "Physical health or characteristic"}


def name(f, dagger=True):
    return NAMES.get(f, f[0].upper() + f[1:]) + (r"$^{\dagger}$" if dagger and f in D["dataset"]["rare6"] else "")


def save(fig, stem):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", metadata={"CreationDate": None, "Producer": None})
    fig.savefig(OUT / f"{stem}.png", dpi=300, metadata={"Software": None})
    plt.close(fig)


def fig2_effects():
    """Paired differences with 95% intervals, one panel per output, identical axes."""
    fig, grid = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(9.6, 4.3), layout="constrained")
    axes = grid.ravel()
    for ax, (metric, title), letter in zip(axes, OUTPUTS, "abcdef"):
        for y, (key, _) in enumerate(CONTRASTS):
            s = D["paired"][key][metric]
            ax.errorbar(s["observed"], y, xerr=[[s["observed"] - s["ci_low"]], [s["ci_high"] - s["observed"]]],
                        fmt="o", color="C0", capsize=2)
        ax.axvline(0, color="0.5", linestyle="--", linewidth=0.8)
        ax.set_xlabel(title)
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.set_title(letter, loc="left", fontweight="bold")
    axes[0].set_yticks(range(len(CONTRASTS)), [label for _, label in CONTRASTS])
    axes[0].set_ylim(len(CONTRASTS) - 0.5, -0.5)
    axes[0].set_xticks([-0.08, -0.04, 0.0, 0.04, 0.08], ["\u22120.08", "", "0", "", "0.08"])
    save(fig, "fig2_effects")


def fig3_factors():
    """Per-factor F1 change from real-only training, factors ordered by support (two-seed means)."""
    c, ds = D["conditions"], D["dataset"]
    order = sorted(ds["factor_support"], key=lambda f: -ds["factor_support"][f])
    style = {"repeated_real": dict(color="0.55", marker="s", mfc="white", label="Repeated real"),
             "sonnet": dict(color="C0", marker="o", label="Sonnet data"),
             "gemma": dict(color="C1", marker="D", label="Gemma data")}
    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(9.6, 3.9), layout="constrained")
    base = c["real_only"]["per_factor_f1_mean_seed"]
    for ax, part, letter in ((axes[0], order[:12], "a"), (axes[1], order[12:], "b")):
        for k, st in style.items():
            ax.plot([c[k]["per_factor_f1_mean_seed"][f] - base[f] for f in part], range(len(part)),
                    linestyle="none", **st)
        ax.axvline(0, color="0.5", linestyle="--", linewidth=0.8)
        ax.set_yticks(range(len(part)), [f"{name(f, False)} ({ds['factor_support'][f]})" + ("*" if f in ds["rare6"] else "")
                                         for f in part])
        ax.set_ylim(len(part) - 0.5, -0.5)
        ax.set_title(letter, loc="left", fontweight="bold")
    fig.supxlabel("F1 change from real-only training (two-seed means)")
    axes[0].set_xticks([-0.2, -0.1, 0.0, 0.1], ["\u22120.2", "\u22120.1", "0", "0.1"])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3)
    save(fig, "fig3_factors")


def tabular(spec, header, rows):
    return "\n".join([f"\\begin{{tabular}}{{{spec}}}", "\\toprule", header + " \\\\", "\\midrule",
                      *rows, "\\bottomrule", "\\end{tabular}"]) + "\n"


def table_conditions():
    """The five conditions and their scores."""
    lines = []
    for label, key in CONDITIONS:
        m = D["conditions"][key]["mean_seed"]
        cells = [f"{m[k]:.4f}" for k in ("composite", "risk_wf1", "phrase_f1", "factor_macro_f1", "rare6_macro_f1")]
        lines.append(" & ".join([label, *cells]) + r" \\")
    (OUT / "table_conditions.tex").write_text(
        tabular("@{}lrrrrr@{}", "Condition & Comp. & Risk & Evid. & Fact. & Six", lines))


def table_labels():
    """Claim C1: the fold model on every generated row and on real rare-factor posts; the human audit."""
    lf, au = D["label_fidelity"], D["audit"]
    rows = [r"\multicolumn{6}{@{}l}{\textit{Out-of-fold baseline model}} \\"]
    for label, key in (("Sonnet posts", "sonnet"), ("Gemma posts", "gemma"), ("Real posts", "real_held_out")):
        x = lf[key]
        rows.append(f"\\quad {label} & {x['n']:,} & {x['risk_agreement']:.3f} & {x['target_recall_balanced']:.3f} & "
                    f"{x['factor_micro']['f1']:.3f} & --" + r" \\")
    rows.append(r"\multicolumn{6}{@{}l}{\textit{Blinded human audit (3-annotator consensus)}} \\")
    for label, key in (("Sonnet posts", "sonnet"), ("Gemma posts", "gemma")):
        w = au["writers"][key]
        target = sum(a for a, _ in w["target_affirmed"].values()) / sum(n for _, n in w["target_affirmed"].values())
        rows.append(f"\\quad {label} & {au['per_writer']} & {w['risk_matches'] / au['per_writer']:.3f} & {target:.3f} & "
                    f"{w['factor_micro']['f1']:.3f} & {w['evidence_phrase_f1']:.3f}" + r" \\")
    (OUT / "table_labels.tex").write_text(tabular(
        "@{}lrrrrr@{}", r"Evaluation set & $n$ & Risk & Target & Fact.\ F\textsubscript{1} & Evid.\ F\textsubscript{1}", rows))


def supplement_tables():
    c, ds = D["conditions"], D["dataset"]
    # S1: per-factor F1
    rows = []
    for f in sorted(ds["factor_support"], key=lambda f: -ds["factor_support"][f]):
        cells = [f"{c[k]['per_factor_f1_mean_seed'][f]:.3f}" for _, k in CONDITIONS]
        rows.append(" & ".join([name(f), str(ds["factor_support"][f]), *cells]) + r" \\")
    (OUT / "tableS1.tex").write_text(tabular(
        "@{}lrrrrrr@{}", "Factor & Support & Real only & Repeated real & Rare-factor weighting & Sonnet data & Gemma data", rows))

    # S2: checkpoint rules and the evidence-free reading
    rob, ck, p = D["robustness"], D["checkpoints"], D["paired"]
    keys = ("sonnet_vs_real_only", "gemma_vs_real_only", "repeated_real_vs_real_only")
    rows = [f"Best on held-out fold (primary) & " + " & ".join(f"{fmt(p[k]['composite']['observed'])} {interval(p[k]['composite'])}" for k in keys) + r" \\",
            f"Final checkpoint (160 updates) & " + " & ".join(f"{fmt(rob['fixed160']['paired'][k]['composite']['observed'])} {interval(rob['fixed160']['paired'][k]['composite'])}" for k in keys) + r" \\",
            f"Best on the other folds & " + " & ".join(f"{fmt(rob['nested']['paired'][k]['composite']['observed'])} {interval(rob['nested']['paired'][k]['composite'])}" for k in keys) + r" \\",
            f"Registered, without evidence & " + " & ".join(f"{fmt(p[k]['composite_no_evidence']['observed'])} {interval(p[k]['composite_no_evidence'])}" for k in keys) + r" \\",
            r"\midrule"]
    for st in ("40", "80", "120", "160"):
        b = ck["paired_by_step"][st]
        rows.append(f"Fixed at {st} updates & " + " & ".join(f"{fmt(b[k]['observed'])} {interval(b[k])}" for k in keys) + r" \\")
    (OUT / "tableS2.tex").write_text(tabular(
        "@{}llll@{}", "Checkpoint rule & Sonnet data & Gemma data & Repeated real", rows))

    # S3: writer pilots
    ps = D["pilot_survey"]
    label = {"haiku": "Claude Haiku 5.5", "sonnet": "Claude Sonnet 5.5", "qwen": "Qwen3.8-27B",
             "hemmingway": "Hemmingway-1", "gemma12": "Gemma-4-12B", "gemma31": "Gemma-4-31B"}
    rows = []
    for arm in ("sonnet", "qwen", "hemmingway", "gemma12", "gemma31"):
        x = ps["v2"][arm]
        rows.append(f"{label[arm]} & {x['passed_checks']} & {x.get('rejects', {}).get('soft_refusal', 0)} & "
                    f"{x['risk_agreement']:.3f} & {x['target_recall']:.3f} & {x['factor_micro_f1']:.3f} & "
                    f"{100 * x['taxonomy_vocabulary']:.0f}\\% & {x['words_mean']:.0f} & {x['realism_auc']:.3f}" + r" \\")
    ref = ps["v1_real_reference"]
    rows.append(r"\midrule")
    rows.append(f"Real posts (folds 1--4) & & & & & & {100 * ref['taxonomy_vocabulary']:.0f}\\% & {ref['words_mean']:.0f} &" + r" \\")
    (OUT / "tableS3.tex").write_text(tabular(
        "@{}lrrrrrrrr@{}", "Writer & Passed & Refused & Risk & Target & Factor F\\textsubscript{1} & "
        "Label words & Words & AUC", rows))

    # S4: per-target recall of the fold model
    lf = D["label_fidelity"]
    rows = [f"{name(f, False)} & {lf['sonnet']['target_recall_by_factor'][f]:.3f} & {lf['gemma']['target_recall_by_factor'][f]:.3f} & "
            f"{lf['real_held_out']['target_recall_by_factor'][f]:.3f} ({lf['real_held_out']['n_by_factor'][f]})" + r" \\"
            for f in ds["rare6"]]
    (OUT / "tableS4.tex").write_text(tabular("@{}lrrr@{}", "Target factor & Sonnet posts & Gemma posts & Real posts (n)", rows))

    # S5: audit target affirmation
    au = D["audit"]["writers"]
    rows = [f"{name(f, False)} & {au['sonnet']['target_affirmed'][f][0]}/{au['sonnet']['target_affirmed'][f][1]} & "
            f"{au['gemma']['target_affirmed'][f][0]}/{au['gemma']['target_affirmed'][f][1]}" + r" \\" for f in ds["rare6"]]
    (OUT / "tableS5.tex").write_text(tabular("@{}lrr@{}", "Target factor & Sonnet & Gemma", rows))

    # S6: the submitted system's per-factor profile
    prof = D["submitted_system"]["factor_profile"]
    order = ds["factor_order"]  # 19 risk factors, then 5 protective

    def line(f):
        x = prof[f]
        return [name(f), str(x["support"]), f"{x['precision']:.3f}", f"{x['recall']:.3f}", f"{x['f1']:.3f}"]
    left = [[r"\multicolumn{5}{@{}l}{\textit{Risk factors}}"]] + [line(f) for f in order[:12]]
    right = ([[r"\multicolumn{5}{@{}l}{\textit{Risk factors, continued}}"]] + [line(f) for f in order[12:19]]
             + [[r"\multicolumn{5}{@{}l}{\textit{Protective factors}}"]] + [line(f) for f in order[19:]])
    left += [["", "", "", "", ""]] * (len(right) - len(left))
    rows = [" & ".join(a + b) + r" \\" for a, b in zip(left, right)]
    head = "Factor & Support & Prec. & Rec. & F\\textsubscript{1} & Factor & Support & Prec. & Rec. & F\\textsubscript{1}"
    spec = r"@{}>{\RaggedRight\arraybackslash}X rrrr @{\hspace{2em}} >{\RaggedRight\arraybackslash}X rrrr@{}"
    body = tabular("SPEC", head, rows).replace("\\begin{tabular}{SPEC}", "\\begin{tabularx}{\\textwidth}{" + spec + "}")
    (OUT / "tableS6.tex").write_text(body.replace("\\end{tabular}", "\\end{tabularx}"))

    # the writer prompts, verbatim
    pr = D["prompts"]
    for stem, text in (("prompt_system_v2", pr["system_v2"]), ("prompt_user", pr["user_layout"])):
        (OUT / f"{stem}.tex").write_text("\\begin{verbatim}\n" + wrap(text) + "\n\\end{verbatim}\n")


def wrap(text, width=110):
    import textwrap
    return "\n".join(textwrap.fill(line, width) if line else "" for line in text.split("\n"))


def figS1_confusion():
    """Risk-level confusion matrix of the submitted system, counts in the cells."""
    ss = D["submitted_system"]
    cm = ss["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(4.4, 3.4), layout="constrained")
    ax.imshow(cm)
    peak = max(max(r) for r in cm)
    for i, row in enumerate(cm):
        for j, v in enumerate(row):
            ax.text(j, i, str(v), ha="center", va="center", color="black" if v > peak / 2 else "white")
    ax.set_xticks(range(4), ss["order"])
    ax.set_yticks(range(4), ss["order"])
    ax.set_xlabel("Predicted level")
    ax.set_ylabel("Annotated level")
    save(fig, "figS1_confusion")


def fmt(x, dp=4):
    s = f"{abs(x):.{dp}f}"
    return f"$-{s}$" if round(x, dp) < 0 else f"$+{s}$" if round(x, dp) > 0 else f"${s}$"


def interval(s, dp=4):
    return f"[{fmt(s['ci_low'], dp)}, {fmt(s['ci_high'], dp)}]"


def pct(x, dp=0):
    return f"{100 * x:.{dp}f}\\%"


NUM = ["no", "one", "two", "three", "four", "five", "six"]


def expected_strings():
    """Every number the paper's prose cites, formatted from frozen.json. A missing string fails the build."""
    p, c, ds, lf, au = D["paired"], D["conditions"], D["dataset"], D["label_fidelity"], D["audit"]
    tf, gen, fb, rob, ck = D["training_files"], D["generation"], D["factor_behaviour"], D["robustness"], D["checkpoints"]
    want = {}

    def cite(label, s):
        want[label] = f"{fmt(s['observed'])} {interval(s)}"

    def lo_hi(s):
        return f"[{fmt(s['ci_low'])}, {fmt(s['ci_high'])}]"

    for key in ("sonnet_vs_repeated_real",
                "gemma_vs_repeated_real", "sonnet_vs_rare_weighted", "gemma_vs_rare_weighted", "sonnet_vs_gemma"):
        cite(f"{key} composite", p[key]["composite"])
    for w in ("sonnet", "gemma"):
        for m in ("risk_wf1", "factor_macro_f1", "phrase_f1", "rare6_macro_f1", "other18_macro_f1", "composite_no_evidence"):
            cite(f"{w} {m}", p[f"{w}_vs_real_only"][m])
        for v in ("fixed160", "nested"):
            cite(f"{w} {v}", rob[v]["paired"][f"{w}_vs_real_only"]["composite"])
        cite(f"{w} step 40", ck["paired_by_step"]["40"][f"{w}_vs_real_only"])
    cite("nested writers", rob["nested"]["paired"]["sonnet_vs_gemma"]["composite"])
    for st in ("80", "120", "160"):
        assert all(ck["paired_by_step"][st][f"{w}_vs_real_only"]["ci_high"] < 0 for w in ("sonnet", "gemma"))
    q = [p["repeated_real_vs_real_only"]["composite"]["observed"]] + \
        [rob[v]["paired"]["repeated_real_vs_real_only"]["composite"]["observed"] for v in ("fixed160", "nested")]
    assert max(abs(x) for x in q) < 0.002
    for key, _ in [("real_only", 0), ("sonnet", 0), ("gemma", 0), ("repeated_real", 0), ("rare_weighted", 0)]:
        want[f"level {key}"] = f"{c[key]['mean_seed']['composite']:.4f}"
    ps_, pg_ = p["sonnet_vs_real_only"]["composite"], p["gemma_vs_real_only"]["composite"]
    want["abstract composite"] = (f"{fmt(ps_['observed'])} (95\\% CI {lo_hi(ps_)}) for Sonnet and "
                                  f"{fmt(pg_['observed'])} ({lo_hi(pg_)}) for Gemma")
    want["results composite"] = f"({fmt(ps_['observed'])}, 95\\% CI {lo_hi(ps_)})"
    pr_ = p["repeated_real_vs_real_only"]["composite"]
    want["repeated_real_vs_real_only composite"] = f"({fmt(pr_['observed'])} vs.\\ real-only, {lo_hi(pr_)})"
    assert all(D["registered"]["holm"][k]["reject_null"] and D["registered"]["holm"][k]["p"] < 0.001 for k in ("P1", "P2"))
    assert not any(a["adopt"] for a in D["registered"]["adoption"].values())
    # label checks
    s, g, r = lf["sonnet"], lf["gemma"], lf["real_held_out"]
    want["risk agreement"] = (f"{pct(s['risk_agreement'], 1)} of Sonnet posts and {pct(g['risk_agreement'], 1)} of Gemma posts, "
                              f"compared to {pct(r['risk_agreement'], 1)}")
    assert min(s["target_recall_balanced"], g["target_recall_balanced"]) > r["target_recall_balanced"]
    want["factor f1"] = (f"({s['factor_micro']['f1']:.3f} and {g['factor_micro']['f1']:.3f} "
                         f"vs.\\ {r['factor_micro']['f1']:.3f})")
    mil = "meaning in life"
    want["meaning"] = (f"{pct(s['target_recall_by_factor'][mil], 1)} of Sonnet posts, {pct(g['target_recall_by_factor'][mil], 1)} "
                       f"of Gemma posts, and {pct(r['target_recall_by_factor'][mil], 1)}")
    want["real n"] = f"{r['n']} held-out real posts"
    want["audit agreement"] = f"{au['risk_all_three_agree']} of the {au['n_posts']} posts (Fleiss' $\\kappa = {au['risk_fleiss_kappa']:.3f}$)"
    want["audit factor kappa"] = f"mean per-factor Fleiss' $\\kappa$ was {au['factor_mean_fleiss_kappa']:.3f}"
    sw, gw = au["writers"]["sonnet"], au["writers"]["gemma"]
    assert sw["risk_matches"] == gw["risk_matches"]
    want["audit risk"] = f"{sw['risk_matches']} of {au['per_writer']} posts ({pct(sw['risk_matches'] / au['per_writer'], 1)}) for both Sonnet and Gemma"
    want["abstract audit"] = f"blinded human annotators agreed on {sw['risk_matches']} of {au['per_writer']} posts per generator"
    pr = au["paired_risk_match"]
    assert pr["sonnet_only"] == pr["gemma_only"]
    want["audit paired"] = f"agreeing on the same {pr['both']} prompts and matching on {pr['sonnet_only']} distinct prompts each"
    want["audit factors"] = f"{sw['factor_micro']['f1']:.3f} for Sonnet and {gw['factor_micro']['f1']:.3f} for Gemma"
    ex, so = "exposure to others' suicide", "sexual orientation related issues"
    want["audit exposure"] = f"(marked in {sw['target_affirmed'][ex][0]} of {sw['target_affirmed'][ex][1]} posts)"
    want["audit meaning"] = f"({sw['target_affirmed'][mil][0]} of {sw['target_affirmed'][mil][1]})"
    want["audit orientation"] = (f"in {gw['target_affirmed'][so][1] - gw['target_affirmed'][so][0]} of "
                                 f"{gw['target_affirmed'][so][1]} posts")
    want["naturalness"] = f"({sw['naturalness_mean']:.2f} vs.\\ {gw['naturalness_mean']:.2f} on a 5-point scale)"
    # generation and files
    gs, gg = gen["sonnet"], gen["gemma"]
    assert gs["cases"] == gg["cases"]
    want["passed"] = f"Out of {gs['cases']:,} candidate prompts per generator across the five folds, {gs['passed_checks']:,} Sonnet posts and {gg['passed_checks']:,} Gemma posts passed"
    assert max(gs["check_rejects"], key=gs["check_rejects"].get) == "plan_disclaimer"
    assert max(gg["check_rejects"], key=gg["check_rejects"].get) == "real_overlap"
    want["sonnet rejects"] = f"``no plan'' ({gs['check_rejects']['plan_disclaimer']} posts)"
    want["gemma rejects"] = f"overlap with real posts ({gg['check_rejects']['real_overlap']} posts)"
    gr, rr = tf["generated_rows_per_fold"], tf["real_rows_per_fold"]
    want["dose"] = f"{min(gr):,} to {max(gr):,} passing posts per fold ({sum(gr):,} total per generator)"
    want["fig dose"] = f"adding {min(gr):,} to {max(gr):,} synthetic posts per split"
    want["real range"] = f"({min(rr):,} to {max(rr):,} posts)"
    want["added"] = f"add an average of {round(sum(gr) / len(gr)):,} posts per fold"
    pos = [v for w in ("sonnet", "gemma") for v in tf["writers"][w]["rare_positives"].values()]
    want["positives"] = f"added {min(pos):,} to {max(pos):,} positive training examples"
    mult = [x for w in ("sonnet", "gemma") for x in tf["writers"][w]["positive_multiple_range"]]
    want["multiple"] = f"{round(min(mult))} to {round(max(mult))} times as many positive training examples"
    passes = tf["passes_over_real_at_160"]
    assert round(passes["real_only"]) == 2 and round(passes["augmented"]) == 1
    # where the loss fell
    base = c["real_only"]["per_factor_f1_mean_seed"]
    other = [f for f in base if f not in ds["rare6"]]
    four = None
    for w in ("sonnet", "gemma"):
        worst = sorted(other, key=lambda f: c[w]["per_factor_f1_mean_seed"][f] - base[f])[:4]
        four = four or set(worst)
        assert set(worst) == four, worst
    for f in ("sense of responsibility", "stressful life event", "coping strategy", "dysfunctional family"):
        assert f in four
        lead = "from " if f == "sense of responsibility" else ""
        tail = " for Sonnet and " if f == "sense of responsibility" else " and "
        gem = " for Gemma" if f == "sense of responsibility" else ""
        want[f] = (f"\\textit{{{f}}} ({lead}{base[f]:.3f} to {c['sonnet']['per_factor_f1_mean_seed'][f]:.3f}{tail}"
                   f"{c['gemma']['per_factor_f1_mean_seed'][f]:.3f}{gem}")
    want["density"] = (f"({fb['sonnet']['factors_per_post']:.2f} and {fb['gemma']['factors_per_post']:.2f} vs.\\ "
                       f"{fb['real_only']['factors_per_post']:.2f})")
    want["precision"] = (f"({fb['sonnet']['micro_precision']:.3f} and {fb['gemma']['micro_precision']:.3f} vs.\\ "
                         f"{fb['real_only']['micro_precision']:.3f})")
    want["real direction"] = f"({pct(r['risk_read_higher'], 1)} over-predicted, {pct(r['risk_read_lower'], 1)} under-predicted)"
    assert s["risk_read_lower"] == g["risk_read_lower"]
    want["generated direction"] = (f"{pct(s['risk_read_higher'], 1)} of Sonnet posts and {pct(g['risk_read_higher'], 1)} of Gemma posts")
    want["generated lower"] = f"lower risk level on only {pct(s['risk_read_lower'], 1)}"
    below = [fb[w]["risk_calls_below_annotated"] for w in ("sonnet", "gemma")]
    assert round(below[0]) == round(below[1])
    want["below"] = f"roughly {round(below[0])} real posts each (vs.\\ {round(fb['real_only']['risk_calls_below_annotated'])} for real-only training)"
    att = lambda k: fb[k]["predicted_risk_share"]["Attempt"]
    want["attempt"] = (f"({pct(att('sonnet'), 1)} and {pct(att('gemma'), 1)} of posts, vs.\\ {pct(att('real_only'), 1)} for "
                       f"real-only and {pct(fb['gold_risk_share']['Attempt'], 1)} in ground truth)")
    sr, gr_ = tf["writers"]["sonnet"], tf["writers"]["gemma"]
    want["words"] = (f"({sr['generated_rows']['words_mean']:.0f} words for Sonnet and {gr_['generated_rows']['words_mean']:.0f} for Gemma, vs.\\ "
                     f"{sr['real_rows']['words_mean']:.0f} words in real data)")
    want["factors per row"] = (f"{sr['generated_rows']['factors_per_row']:.2f} (Sonnet) and {gr_['generated_rows']['factors_per_row']:.2f} "
                               f"(Gemma) positive factors per post, compared to {sr['real_rows']['factors_per_row']:.2f}")
    pv = D["prevalence_vs_f1"]
    want["spearman"] = f"$\\rho = {pv['sonnet']['spearman_rho']:.2f}$ for Sonnet and ${pv['gemma']['spearman_rho']:.2f}$ for Gemma"
    # data and setup facts
    want["posts users"] = f"{ds['n_posts']:,} annotated posts from {ds['n_users']} users"
    rc = ds["risk_counts"]
    want["risk counts"] = (f"{rc['Indicator']} \\textit{{Indicator}}, {rc['Ideation']} \\textit{{Ideation}}, "
                           f"{rc['Behavior']} \\textit{{Behavior}}, and {rc['Attempt']} \\textit{{Attempt}}")
    want["distinct"] = f"{ds['n_distinct_inputs']:,} unique inputs"
    want["no factor"] = f"{ds['n_no_factor']} posts have no annotated factors"
    want["indicator evidence"] = f"{ds['indicator_posts_with_evidence']} \\textit{{Indicator}} posts in the dataset include annotated phrases"
    fs = ds["factor_support"]
    want["support range"] = f"{max(fs.values())} posts (\\textit{{hopelessness}}) down to {min(fs.values())}"
    others = [fs[f] for f in fs if f not in ds["rare6"]]
    want["other supports"] = f"(versus {min(others)} to {max(others)} for the other 18)"
    rare = [fs[f] for f in ds["rare6"]]
    want["rare range"] = f"six of which have only {min(rare)} to {max(rare)} positive examples"
    t = lambda x: f"\\textit{{{x}}}"
    want["rare supports"] = (f"{t('cognitive deficits')}, {fs['cognitive deficits']} posts; {t('substance use')}, {fs['substance use']}; "
                             f"{t('poor school performance')}, {fs['poor school performance']}; {t(ex)}, "
                             f"{fs[ex]}; {t(so)}, {fs[so]}) and the rarest protective factor "
                             f"({t(mil)}, {fs[mil]})")
    fu = ds["folds_users"]
    want["fold users"] = f"{min(fu)} to {max(fu)} users"
    want["leaderboard"] = f"{D['public_leaderboard_real_only_ensemble']['composite']:.4f} on the public leaderboard"
    want["meaning oracle"] = f"of at most {D['meaning_in_life_oracle_cutoff_f1']:.2f} on it"
    return want


def supplement_strings():
    """Numbers the supplement's prose cites (its tables are generated, so only the prose is checked)."""
    c, ds, ps, tf, lf, au = D["conditions"], D["dataset"], D["pilot_survey"], D["training_files"], D["label_fidelity"], D["audit"]
    base = c["real_only"]["per_factor_f1_mean_seed"]
    opposite = sum((c["sonnet"]["per_factor_f1_mean_seed"][f] - base[f]) * (c["gemma"]["per_factor_f1_mean_seed"][f] - base[f]) < 0
                   for f in ds["rare6"])
    sel = D["checkpoints"]["selected_steps"]
    late = sum(s in (120, 160) for v in sel.values() for s in v)
    open_w = [ps["v2"][a]["taxonomy_vocabulary"] for a in ("qwen", "hemmingway", "gemma12", "gemma31")]
    aucs = [ps["v2"][a]["realism_auc"] for a in ("sonnet", "qwen", "hemmingway", "gemma12", "gemma31")]
    best_open = max(("qwen", "hemmingway", "gemma12", "gemma31"), key=lambda a: ps["v2"][a]["risk_agreement"])
    assert best_open == "gemma31"
    assert max(ps["v1"], key=lambda a: ps["v1"][a]["risk_agreement"]) == "sonnet" == max(ps["v2"], key=lambda a: ps["v2"][a]["risk_agreement"])
    sr, gr = tf["writers"]["sonnet"], tf["writers"]["gemma"]
    r, s, g = lf["real_held_out"], lf["sonnet"], lf["gemma"]
    mism = [k for w in au["writers"].values() for k, n in w["risk_mismatches"].items() for _ in range(n)]
    assert all("Behavior" in k for k in mism)
    pw = D["public_leaderboard_real_only_ensemble"]
    ov = D["prompts"]["overrides_v2"]
    return {
        "opposite": f"on {NUM[opposite]} of the six",
        "late": f"Across the {sum(len(v) for v in sel.values())} runs of the four conditions shown, step 120 or step 160 was selected in {late} runs",
        "haiku": f"{ps['v1']['haiku']['rejects']['soft_refusal']} of the 300 pilot prompts",
        "label words": (f"in {100 * min(open_w):.0f}\\% to {100 * max(open_w):.0f}\\% of its generated posts, compared to "
                        f"{100 * ps['v2']['sonnet']['taxonomy_vocabulary']:.0f}\\% for Claude Sonnet 5.5 and "
                        f"{100 * ps['v1_real_reference']['taxonomy_vocabulary']:.0f}\\% in real training posts"),
        "auc": f"(ROC-AUC {min(aucs):.3f} to {max(aucs):.3f})",
        "candidates": f"{NUM[len({a for v in ('v1', 'v2') for a in ps[v] if a != 'v1_real_reference'})]} candidate generator models",
        "sonnet pilot": f"highest risk-level agreement ({ps['v2']['sonnet']['risk_agreement']:.3f})",
        "gemma pilot": f"among the open-weights models ({ps['v2']['gemma31']['risk_agreement']:.3f})",
        "overrides": " ".join(f"(``{ov[f]}'')" for f in ("meaning in life",)),
        "words": (f"{sr['generated_rows']['words_mean']:.0f} words for Sonnet and {gr['generated_rows']['words_mean']:.0f} for Gemma, "
                  f"vs.\\ {sr['real_rows']['words_mean']:.0f} words for real posts"),
        "factors": (f"({sr['generated_rows']['factors_per_row']:.2f} and {gr['generated_rows']['factors_per_row']:.2f} "
                    f"vs.\\ {sr['real_rows']['factors_per_row']:.2f})"),
        "rare carriers": f"carrier posts average {ds['factors_per_post_with_rare_factor']:.2f} factors",
        "attempt": (f"({pct(sr['generated_rows']['risk_share']['Attempt'], 1)} for Sonnet and {pct(gr['generated_rows']['risk_share']['Attempt'], 1)} "
                    f"for Gemma, vs.\\ {pct(sr['real_rows']['risk_share']['Attempt'], 1)} in real data)"),
        "evidence": (f"{sr['generated_rows']['spans_per_evidence_row']:.2f} and {gr['generated_rows']['spans_per_evidence_row']:.2f} "
                     f"phrases of {sr['generated_rows']['tokens_per_span_mean']:.1f} and {gr['generated_rows']['tokens_per_span_mean']:.1f} "
                     f"words for Sonnet and Gemma, vs.\\ {sr['real_rows']['spans_per_evidence_row']:.2f} phrases of "
                     f"{sr['real_rows']['tokens_per_span_mean']:.1f} words"),
        "real direction": f"({pct(r['risk_read_higher'], 1)} predicted higher than annotated, {pct(r['risk_read_lower'], 1)} lower)",
        "generated direction": (f"({pct(s['risk_read_higher'], 1)} of Sonnet posts and {pct(g['risk_read_higher'], 1)} of Gemma posts "
                                f"predicted higher than assigned, vs.\\ {pct(s['risk_read_lower'], 1)} lower)"),
        "seed": f"seed {au['seed']}",
        "agree": f"exact risk level for {au['risk_all_three_agree']} of the {au['n_posts']} posts",
        "pairwise": (f"ranged from {au['pairwise_risk_exact'][0]:.3f} to {au['pairwise_risk_exact'][1]:.3f}, and pairwise factor "
                     f"micro-F\\textsubscript{{1}} ranged from {au['pairwise_factor_micro_f1'][0]:.3f} to {au['pairwise_factor_micro_f1'][1]:.3f}"),
        "majority": f"The remaining {au['majority_decided']} posts were resolved by 2-to-1 majority vote",
        "mismatches": f"All {len(mism)} disagreements between the human consensus risk level",
        "leaderboard": f"composite score of {pw['composite']:.4f}",
        "subtasks": f"({pw['subtask1']:.4f} combined on risk level and evidence, and {pw['subtask2']:.4f} on factors)",
    }


def check_text():
    for path, want in ((MAIN, expected_strings()), (SUPP, supplement_strings())):
        if not path.exists():
            print(f"skipped the text check: put the manuscript's {path.name} in {path.parent}")
            continue
        text = path.read_text()
        missing = {k: v for k, v in want.items() if v not in text}
        if missing:
            sys.exit(f"{path.name} disagrees with frozen.json (or omits these):\n" +
                     "\n".join(f"  {k}: {v}" for k, v in missing.items()))
        print(f"checked {len(want)} numbers against {path.name}")


def fig1_design():
    """Schematic (TikZ), compiled with pdflatex at print size; timestamps stripped for byte-identical rebuilds."""
    env = dict(os.environ, SOURCE_DATE_EPOCH="0", FORCE_SOURCE_DATE="1")
    with tempfile.TemporaryDirectory() as tmp:
        run = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory", tmp,
                              str(HERE / "fig1_design.tex")], env=env, capture_output=True, text=True)
        if run.returncode:
            sys.exit(f"pdflatex failed on fig1_design.tex:\n{run.stdout[-2000:]}")
        (OUT / "fig1_design.pdf").write_bytes((pathlib.Path(tmp) / "fig1_design.pdf").read_bytes())
    subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", OUT / "fig1_design.pdf", OUT / "fig1_design"], check=True)


def build():
    OUT.mkdir(exist_ok=True)
    for stale in ("fig3_exposure", "fig4_phrases"):
        for ext in ("pdf", "png"):
            (OUT / f"{stale}.{ext}").unlink(missing_ok=True)
    for stale in ("table1.tex",):
        (OUT / stale).unlink(missing_ok=True)
    fig1_design()
    fig2_effects()
    fig3_factors()
    table_conditions()
    table_labels()
    supplement_tables()
    figS1_confusion()


def digest():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob("*")) if p.is_file()}


if __name__ == "__main__":
    build()
    if "--check" in sys.argv:
        first = digest()
        build()
        if digest() != first:
            sys.exit("outputs differ between two builds")
        print("two builds byte-identical")
    if "--no-text" not in sys.argv:
        check_text()
