#!/usr/bin/env python3
"""Re-run the four-wheel secondary evaluation on the locked seeds 501-508.

The frozen CSV in data/four_wheel/ is NEVER modified. Re-runs write to
results/ and can be compared against the frozen data row by row
(deterministic seeds should reproduce it exactly).
"""
from dataclasses import replace
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

import pandas as pd

from param_sindy.config import set_params
from param_sindy.study.four_wheel_validation import (
    FW_VERSION,
    build_fourwheel_scenarios,
    one,
    rls_one,
)

SCENES = build_fourwheel_scenarios()
P = replace(
    set_params(),
    mpc=replace(set_params().mpc, horizon=5, n_candidates=16, clearance_margin=0.45),
    sindy=replace(set_params().sindy, window_size=60, update_freq=5, first_update=15),
)
METHODS = ("sindy", "standard", "rls")


def worker(seed):
    rows = []
    for sid, sc in enumerate(SCENES, 1):
        for m in METHODS:
            r = rls_one(sc, P, seed) if m == "rls" else one(sc, P, seed, m)
            rows.append({"seed": seed, "scenario_id": sid, "scenario_name": sc.name,
                         "method": m, **r, "protocol_version": FW_VERSION})
    return rows


def parse_seeds(text):
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in text.split(",") if s.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="501-508", help="e.g. 501-508 or 501,502,507")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="results/four_wheel_rerun.csv")
    ap.add_argument("--compare", action="store_true",
                    help="compare the re-run with the frozen CSV row by row")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(worker, s) for s in seeds]
        for i, f in enumerate(as_completed(futs), 1):
            rows.extend(f.result())
            print(f"[four-wheel] {i}/{len(seeds)} seeds done", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["seed", "scenario_id", "method"]).reset_index(drop=True)
    df.to_csv(out, index=False)

    print("\n=== four-wheel re-run success rates ===")
    for m in METHODS:
        rate = df.loc[df.method == m, "success"].mean()
        print(f"  {m:<10} {rate * 100:5.1f}%")
    print(f"\nwrote {out} ({len(df)} rows)")

    if args.compare:
        frozen = Path(__file__).resolve().parent / "data" / "four_wheel" / "fourwheel.csv"
        fr = pd.read_csv(frozen)
        key = ["seed", "scenario_id", "method"]
        merged = df.merge(fr, on=key, suffixes=("_new", "_frz"))
        num_cols = ["terminal_error", "min_dist", "control_energy", "arrival_time",
                    "steps", "alpha_mean", "alpha_std", "alpha_response",
                    "regime_mean", "residual_mean", "accepted", "rejected"]
        ok = merged.success_new == merged.success_frz
        for c in num_cols:
            a, b = merged[f"{c}_new"], merged[f"{c}_frz"]
            ok &= (a.isna() & b.isna()) | (a - b).abs().le(1e-6)
        agree = int(ok.sum())
        n = len(merged)
        print(f"\nrow agreement with {frozen.name}: {agree}/{n} ({100 * agree / n:.1f}%)")


if __name__ == "__main__":
    main()
