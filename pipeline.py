#!/usr/bin/env python3
"""
pipeline.py — Run the full extraction pipeline (or parts of it).

    python pipeline.py                     # full pipeline for new seeds only
    python pipeline.py --all               # full pipeline for everything
    python pipeline.py --from extract      # start from extraction step
    python pipeline.py --id moral_elevation  # single mechanism
    python pipeline.py --fix-only          # just fix flagged + rebuild
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
PYTHON = str(ROOT / ".venv" / "bin" / "python")


def run(cmd: list[str], label: str, check: bool = True) -> bool:
    """Run a command, print its output live, return success."""
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}\n")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n✗ FAILED: {label} (exit {result.returncode})")
        if check:
            sys.exit(result.returncode)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Run the drivermap extraction pipeline")
    parser.add_argument("--id", help="Process a single mechanism")
    parser.add_argument("--domain", help="Process one domain only")
    parser.add_argument("--all", action="store_true", help="Process all mechanisms (not just new)")
    parser.add_argument(
        "--from",
        dest="start_from",
        choices=["fetch", "extract", "verify", "fix", "narrative", "load"],
        default="fetch",
        help="Start from this step (default: fetch)",
    )
    parser.add_argument(
        "--fix-only", action="store_true", help="Just fix flagged records + rebuild"
    )
    parser.add_argument(
        "--no-narrative", action="store_true", help="Skip narrative_outputs patching"
    )
    parser.add_argument(
        "--fix-rounds", type=int, default=2, help="Max fix→verify rounds (default 2)"
    )
    args = parser.parse_args()

    skip = [] if args.all else ["--skip-existing"]
    target = []
    if args.id:
        target = ["--id", args.id]
    elif args.domain:
        target = ["--domain", args.domain]

    steps = ["fetch", "extract", "verify", "fix", "narrative", "load"]
    start_idx = steps.index(args.start_from) if not args.fix_only else steps.index("fix")

    # ── Fetch
    if start_idx <= 0:
        run([PYTHON, "fetch.py"] + skip + target, "Fetching corpus")

    # ── Extract
    if start_idx <= 1:
        run([PYTHON, "extract.py", "--guided"] + skip + target, "Guided extraction")

    # ── Verify → Fix loop
    if start_idx <= 3:
        for round_n in range(1, args.fix_rounds + 1):
            run(
                [PYTHON, "extract.py", "--verify"] + target,
                f"Verification (round {round_n})",
            )

            # Try fixing flagged records
            ok = run(
                [PYTHON, "extract.py", "--fix"] + target,
                f"Fixing flagged records (round {round_n})",
                check=False,
            )
            if not ok:
                print("  (no records to fix or fix failed — moving on)")
                break

    # ── Narrative outputs
    if start_idx <= 4 and not args.no_narrative:
        narr_args = [PYTHON, "patch_narrative_outputs.py", "--skip-existing"]
        if args.id:
            narr_args = [PYTHON, "patch_narrative_outputs.py", "--id", args.id]
        run(narr_args, "Patching narrative outputs")

    # ── Load + build
    if start_idx <= 5:
        run([PYTHON, "db_load.py", "--rebuild"], "Rebuilding database")
        run([PYTHON, "build_explorer.py"], "Building explorer")

    # ── Tests
    run([ROOT / ".venv" / "bin" / "pytest", "tests/", "-q"], "Running tests", check=False)

    print(f"\n{'═' * 60}")
    print("  Pipeline complete")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
