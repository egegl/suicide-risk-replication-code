"""Training files and run specs of the paper's three real-data conditions.

    python scripts/e2_controls.py build    # repeated-real and rare-weighting files, checked against the paper's
    python scripts/e2_controls.py specs    # runs31/specs.json (real only, repeated real), runs31-rare/specs.json

Run after build_dataset.py. Real only trains on data/processed/sft as is (arm `baseline`). Repeated real
(arm `qcontrol`) repeats real rows up to real + pkgc.FULL_DOSE rows per fold. Rare-factor weighting (arm
`rareweight`) multiplies the loss weight by 3 on real rows with any of the five rare risk factors
(meaning in life is excluded). Run ids keep the paper's names because they seed the training order.
Train and evaluate with `python run_experiment.py --runs-dir runs31 train --index N` (and runs31-rare).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import random  # noqa: E402

from synth import pkgc, state  # noqa: E402
from synth import constants as SC  # noqa: E402
from train import constants as TC  # noqa: E402
from train.specs import _mean_weight  # noqa: E402

SFT = ROOT / "data/processed/sft"
QCONTROL = ROOT / "data/synthetic/arms/qcontrol"
RARE = ROOT / "data/processed/sft_rareweight"
RARE_W = 3.0
RARE_TARGETS = tuple(f for f in SC.RARE_FACTORS if f != "meaning in life")
PAPER_SHA256 = {  # the training files behind the paper
    "qcontrol": ("d01e1c9e202004fc75fcc20a41b73eeaf96c826a0cc6ae7331ad7740129c456c",
                 "59ff707fa6325479bb7f8980cbefead5d5bbe36052c0f89bf898990ab5f1fe36",
                 "40fb0fc0ed5c1ef3578a386c8c5bc6edaa13dc80fb78f265e34591677bd9dc38",
                 "a582d6381ba20f7d184d3589ce073c65c1d0ab614f4d762d9ef758156bb4cd98",
                 "858cc8325b84bd0a59d831f140e2d0fff43dc71a76c3842fa1aa4ffa624d2cdd"),
    "rareweight": ("549e0910f37721e8daa2f5b9c7a38a7f5bd8aa61aeb16d129d534dc9b05a8b60",
                   "31901d3f0623542f4bafeabe03be9a4b4d33c337f45c373bb0c7ee7ec63c78b6",
                   "41ba41b0b0136f573bb4e30b6415e378c71a903ad414c31a98f6d2ae9c8fb2e9",
                   "d68e886aeb7d8c4de7b30db348078813843dabaf1b3a393319f37959c940439d",
                   "86ce02ff14c67c8f65483019b19e349f0ae8a932cc5aacb0e360ad22c5666b59"),
}


def _upsample(real: list[dict], target: int, seed_key: str) -> list[dict]:
    """The competition sweep's repeated-real recipe: whole copies, then a seeded remainder."""
    out = list(real) * (target // len(real))
    extra = list(real)
    random.Random(seed_key).shuffle(extra)
    return out + extra[:target - len(out)]


def build() -> None:
    manifest = {"arm": "qcontrol", "folds": {}}
    for k in range(5):
        real = state.read_jsonl(SFT / f"train_fold{k}.jsonl")
        rows = _upsample(real, len(real) + pkgc.FULL_DOSE[k], f"{SC.SYNTH_SEED}|qcontrol|{k}")
        random.Random(f"{SC.SYNTH_SEED}|assemble|qcontrol|{k}").shuffle(rows)
        state.atomic_write_jsonl(QCONTROL / f"train_fold{k}.jsonl", rows)
        manifest["folds"][str(k)] = {"n_real": len(rows), "sha256_out": state.sha256_file(QCONTROL / f"train_fold{k}.jsonl")}
        rare = [{**r, "weight": float(r["weight"]) * RARE_W}
                if any(json.loads(r["target"])["factors"][f] for f in RARE_TARGETS) else dict(r) for r in real]
        state.atomic_write_jsonl(RARE / f"train_fold{k}.jsonl", rare)
        for arm, path in (("qcontrol", QCONTROL), ("rareweight", RARE)):
            got = state.sha256_file(path / f"train_fold{k}.jsonl")
            print(arm, f"fold{k}", "matches the paper" if got == PAPER_SHA256[arm][k] else f"DIFFERS ({got[:12]})")
    state.atomic_write_json(QCONTROL / "manifest.json", manifest)


def specs() -> None:
    def spec(arm, k, seed, train_rel, index):
        path = ROOT / train_rel
        return {"run_id": f"{arm}-f{k}-s{seed}", "index": index, "arm": arm, "fold": k, "seed": seed,
                "train_file": train_rel, "train_file_sha": state.sha256_file(path),
                "mean_weight": _mean_weight(path), "val_file": f"{TC.SFT_DIR}/val_fold{k}.jsonl",
                "base_model": TC.BASE_MODEL, "total_steps": TC.TOTAL_STEPS,
                "checkpoint_steps": list(TC.CHECKPOINT_STEPS)}
    for runs, arms in (("runs31", {"baseline": SFT, "qcontrol": QCONTROL}), ("runs31-rare", {"rareweight": RARE})):
        out = []
        for arm, d in arms.items():
            for k in range(5):
                for seed in (1, 2):
                    out.append(spec(arm, k, seed, str((d / f"train_fold{k}.jsonl").relative_to(ROOT)), len(out)))
        state.atomic_write_json(ROOT / runs / "specs.json", out)
        print(runs, len(out), "specs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("build", "specs"))
    {"build": build, "specs": specs}[ap.parse_args().cmd]()
