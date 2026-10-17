"""E2 confirmatory analysis (registered): the ONE analysis the E2 claims cite.
Written and committed before any E2 evaluation existed.

Inputs: pooled, nested-thresholded OOF predictions per arm and seed
(<runs>/pooled/<arm>-s<k>.thresholded.jsonl, from `run_experiment.py pool` and
`run_experiment.py thresholds` with the default argmax estimator, as for runs31).

Reproduction guard: the three reused controls must reproduce, per seed, the
composites frozen for the ICHI paper (paper/figures/frozen.json, stored to 8
decimals) or the script aborts.

Registered readout:
  P1 = pkgc_sonnet - real_only, P2 = pkgc_gemma - real_only on the mean-of-two-
  seeds pooled composite; paired user bootstrap with the registered shared draws
  (B=2000, seed 20260815); two-sided bootstrap p = 2 min(P(d<=0), P(d>=0));
  Holm over {P1, P2} at alpha 0.05.
  Adoption (D3, per arm): bootstrap-mean delta vs real_only >= SHIP_BAR (0.01)
  AND bootstrap-mean delta vs repeated_real > 0.
  Secondary (reported, not tested): each arm vs repeated_real and vs
  rare_weighted, sonnet - gemma, components, rare-6 macro, per-factor F1.
  --seed1-only applies the registered compute fallback (seed-1 pools only).

Writes runs31-pkgc/reanalysis/e2_confirm.json.
Usage: python scripts/e2_confirm.py [--seed1-only]
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location(
    "freeze", ROOT / "paper/figures/freeze.py")
freeze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(freeze)            # main() is guarded; this only defines helpers

import scorer                               # noqa: E402
from prep.constants import FACTORS_24       # noqa: E402
from train import oof                       # noqa: E402
from train.constants import SHIP_BAR        # noqa: E402

CONTROLS = {"real_only": ("runs31", "baseline"),
            "repeated_real": ("runs31", "qcontrol"),
            "rare_weighted": ("runs31-rare", "rareweight")}
ARMS = {"pkgc_sonnet": ("runs31-pkgc", "pkgc_sonnet"),
        "pkgc_gemma": ("runs31-pkgc", "pkgc_gemma")}
PRIMARY = {"P1": ("pkgc_sonnet", "real_only"), "P2": ("pkgc_gemma", "real_only")}
SECONDARY = {"sonnet_vs_repeated_real": ("pkgc_sonnet", "repeated_real"),
             "gemma_vs_repeated_real": ("pkgc_gemma", "repeated_real"),
             "sonnet_vs_rare_weighted": ("pkgc_sonnet", "rare_weighted"),
             "gemma_vs_rare_weighted": ("pkgc_gemma", "rare_weighted"),
             "sonnet_vs_gemma": ("pkgc_sonnet", "pkgc_gemma")}
ALPHA = 0.05
OUT = ROOT / "runs31-pkgc/reanalysis/e2_confirm.json"


def two_sided_p(d: np.ndarray) -> float:
    return float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))


def holm(pvals: dict) -> dict:
    """Holm step-down: reject in ascending-p order while p <= alpha/(m-rank)."""
    order = sorted(pvals, key=pvals.get)
    out, stop = {}, False
    for rank, key in enumerate(order):
        thresh = ALPHA / (len(order) - rank)
        reject = (not stop) and pvals[key] <= thresh
        stop = stop or not reject
        out[key] = {"p": pvals[key], "holm_threshold": thresh, "reject_null": reject}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed1-only", action="store_true", help="registered compute fallback")
    args = ap.parse_args()
    seeds = (1,) if args.seed1_only else (1, 2)

    gold = scorer.load_gold(freeze.use("data/processed/gold.jsonl"))
    posts = pd.read_parquet(freeze.use("data/processed/posts.parquet"), columns=["row_id", "anon_user_id"])
    users = dict(zip(posts.row_id, posts.anon_user_id))
    ids = sorted(gold)
    sc = freeze.Scorer(gold, ids)
    frozen = json.loads((ROOT / "paper/figures/frozen.json").read_text())

    labs, per_seed = {}, {}
    for name, (runs, arm) in {**CONTROLS, **ARMS}.items():
        for s in seeds:
            rows = freeze.read_jsonl(f"{runs}/pooled/{arm}-s{s}.thresholded.jsonl")
            lab = sc.label(oof.propagate(freeze.preds(rows), freeze.P))
            labs[(name, s)] = lab
            got = sc.score(lab, np.ones(len(ids)))
            if name in CONTROLS:   # reproduction guard
                want = frozen["conditions"][name]["per_seed"][str(s)]["composite"]
                freeze.expect(f"{name} s{s} composite vs frozen.json", round(got["composite"], 8), want, 1e-8)
            per_seed.setdefault(name, {})[str(s)] = {
                **{k: got[k] for k in freeze.METRICS},
                "per_factor_f1": dict(zip(FACTORS_24, got["per_factor_f1"].tolist()))}

    W, n_users = freeze.draws(gold, users, ids)
    reps = freeze.replicate_table(sc, {f"{n}-s{s}": labs[(n, s)] for (n, s) in labs}, W)
    names = list(CONTROLS) + list(ARMS)
    arm_reps = {n: {k: np.mean([reps[f"{n}-s{s}"][k] for s in seeds], axis=0) for k in freeze.METRICS}
                for n in names}
    observed = {n: {k: float(np.mean([per_seed[n][str(s)][k] for s in seeds])) for k in freeze.METRICS}
                for n in names}

    def contrast(t: str, r: str) -> dict:
        out = {}
        for k in freeze.METRICS:
            d = arm_reps[t][k] - arm_reps[r][k]
            out[k] = {**freeze.summarize(d), "observed": observed[t][k] - observed[r][k],
                      "p_two_sided": two_sided_p(d)}
        return out

    primary = {key: contrast(*pair) for key, pair in PRIMARY.items()}
    holm_result = holm({key: primary[key]["composite"]["p_two_sided"] for key in PRIMARY})
    adoption = {}
    for arm in ARMS:
        d_base = float(np.mean(arm_reps[arm]["composite"] - arm_reps["real_only"]["composite"]))
        d_q = float(np.mean(arm_reps[arm]["composite"] - arm_reps["repeated_real"]["composite"]))
        adoption[arm] = {"delta_vs_real_only": d_base, "delta_vs_repeated_real": d_q,
                         "ship_bar": SHIP_BAR, "adopt": bool(d_base >= SHIP_BAR and d_q > 0)}

    result = {
        "registration": "E2 (registered analysis)", "seeds": list(seeds),
        "fallback_seed1_only": args.seed1_only, "bootstrap_B": len(W), "n_users": n_users,
        "controls_reproduced": True,
        "observed_mean_seed": observed, "per_seed": per_seed,
        "primary": primary, "holm": holm_result, "adoption": adoption,
        "secondary": {key: contrast(*pair) for key, pair in SECONDARY.items()},
        "inputs_sha256": freeze.INPUTS,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(freeze.clean(result), indent=2) + "\n")
    for key, (t, r) in PRIMARY.items():
        c = primary[key]["composite"]
        print(f"{key} {t} - {r}: {c['observed']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] "
              f"p={c['p_two_sided']:.4f} holm_reject={holm_result[key]['reject_null']} "
              f"adopt={adoption[t]['adopt']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
