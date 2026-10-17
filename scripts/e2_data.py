"""CPU steps of the generated-data pipeline (synth.pkgc), run from the repo root.

    python scripts/e2_data.py prepare                    # 300-case fold-0 pilot (pkgc-v2)
    python scripts/e2_data.py prepare --fold 0           # E2 cases and prompts for held-out fold 0 (repeat for 1-4)
    python scripts/e2_data.py report                     # pilot report (supplement Table S3)
    python scripts/e2_data.py assemble --arm sonnet      # E2 training files (and --arm gemma31)
    python scripts/e2_data.py specs                      # runs31-pkgc/specs.json (20 runs)
    python scripts/e2_data.py audit-packet               # blinded 48-post audit packet and key
    python scripts/e2_data.py audit-consensus            # 2-of-3 consensus (reads no key)
    python scripts/e2_data.py audit-report               # unblinded agreement report

The writers (scripts/pkgc_claude.py, scripts/pkgc_local.py) run between prepare and assemble.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from synth import pkgc  # noqa: E402

PROCESSED = Path("data/processed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", choices=sorted(pkgc.ROOTS), default=pkgc.PROMPT_VERSION)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare").add_argument("--fold", type=int, choices=range(5))
    sub.add_parser("report")
    sub.add_parser("assemble").add_argument("--arm", choices=("sonnet", "gemma31"), required=True)
    for name in ("specs", "audit-packet", "audit-consensus", "audit-report"):
        sub.add_parser(name)
    args = ap.parse_args()
    os.chdir(ROOT)

    if args.cmd == "prepare":
        out = pkgc.prepare(PROCESSED, args.version, args.fold)
    elif args.cmd == "report":
        out = pkgc.report(PROCESSED, pkgc.ROOTS[args.version])["arms"]
    elif args.cmd == "assemble":
        out = pkgc.assemble(args.arm, args.version)
    elif args.cmd == "specs":
        out = [s["run_id"] for s in pkgc.write_e2_specs()]
    elif args.cmd == "audit-packet":
        out = pkgc.build_audit_packet()
    elif args.cmd == "audit-consensus":
        out = {"cases": len(pkgc.audit_consensus()["responses"])}
    else:
        out = pkgc.audit_report()["agreement"]
    print(json.dumps(out, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
