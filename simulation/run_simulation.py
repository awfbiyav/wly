#!/usr/bin/env python3
"""One-command simulation pipeline over the frozen data.

    python run_simulation.py

Step 1  audit    - recompute the statistics from the three locked CSVs and
                   check the pre-registered quality gates   (audit.py)
Step 2  figures  - render the headline result figures          (make_figures.py)

Option:
    --rerun-four-wheel   additionally re-run the four-wheel evaluation on the
                         locked seeds 501-508 into results/ and compare it
                         row by row with the frozen CSV (frozen data are
                         never modified)
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(cmd):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=str(ROOT))
    if r.returncode:
        raise SystemExit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun-four-wheel", action="store_true")
    args = ap.parse_args()
    run([PY, "audit.py"])
    run([PY, "make_figures.py"])
    if args.rerun_four_wheel:
        run([PY, "run_four_wheel.py", "--compare"])


if __name__ == "__main__":
    main()
