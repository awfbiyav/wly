#!/usr/bin/env python3
"""Reproducibility audit over the frozen simulation results.

Reads the three locked CSVs (never modified by this script):

    data/main/main.csv                         300-episode main benchmark
    data/prediction/prediction_benchmark.csv   post-shift prediction RMSE
    data/four_wheel/fourwheel.csv              four-wheel secondary validation

Recomputes the headline statistics, checks the pre-registered quality gates,
writes results/audit.json, and prints a summary. Exit 0 = all gates pass.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
MAIN = ROOT / "data" / "main" / "main.csv"
PRED = ROOT / "data" / "prediction" / "prediction_benchmark.csv"
FW = ROOT / "data" / "four_wheel" / "fourwheel.csv"
OUT = ROOT / "results" / "audit.json"
MAIN_SHA256 = "9155a0d85c7659d41da7ab70c44ca840a65965d7e0286918f146e0e87cc70748"
METHODS = ("sindy", "standard", "rls")


def sha256(path):
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hh = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - hh, c + hh


def mcnemar_exact(wins, losses):
    from math import comb

    disc = wins + losses
    if disc == 0:
        return 1.0
    p = 0.0
    for k in range(disc + 1):
        if abs(2 * k - disc) >= abs(2 * wins - disc):
            p += comb(disc, k) / (2 ** disc)
    return float(min(1.0, p))


def main():
    main_df = pd.read_csv(MAIN)
    pred = pd.read_csv(PRED)
    fw = pd.read_csv(FW)

    out = {
        "version": "1.0",
        "provenance": {
            "main_sha256": sha256(MAIN),
            "expected": MAIN_SHA256,
            "main_unchanged": sha256(MAIN) == MAIN_SHA256,
            "main_rows": len(main_df),
        },
        "main": {},
        "prediction": {},
        "four_wheel": {},
    }

    ms = []
    for m in METHODS:
        d = main_df[main_df.method == m]
        k = int(d.success.sum())
        lo, hi = wilson(k, len(d))
        ms.append({
            "method": m, "success": k / len(d), "n": len(d),
            "ci_low": lo, "ci_high": hi,
            "terminal_error_mean": d.terminal_error.mean(),
            "min_dist_mean": d.min_dist.mean(),
        })
    out["main"] = {"summary": ms}

    out["prediction"] = {
        "n": len(pred),
        "sindy_rmse": float(pred.sindy_rmse.mean()),
        "nominal_rmse": float(pred.nominal_rmse.mean()),
        "rls_rmse": float(pred.rls_rmse.mean()),
    }
    out["prediction"]["gain_vs_nominal_pct"] = 100 * (
        1 - out["prediction"]["sindy_rmse"] / out["prediction"]["nominal_rmse"])
    out["prediction"]["gain_vs_rls_pct"] = 100 * (
        1 - out["prediction"]["sindy_rmse"] / out["prediction"]["rls_rmse"])

    fs = []
    for m, d in fw.groupby("method"):
        k = int(d.success.sum())
        lo, hi = wilson(k, len(d))
        fs.append({
            "method": m, "success": k / len(d), "n": len(d),
            "ci_low": lo, "ci_high": hi,
            "terminal_error_mean": d.terminal_error.mean(),
            "min_dist_mean": d.min_dist.mean(),
            "energy_mean": d.control_energy.mean(),
        })
    piv = fw.pivot_table(index=["seed", "scenario_name"], columns="method",
                         values="success", aggfunc="first").dropna()
    paired = {}
    for b in ("standard", "rls"):
        a = piv.sindy
        wins = int(((a == 1) & (piv[b] == 0)).sum())
        losses = int(((a == 0) & (piv[b] == 1)).sum())
        paired[b] = {
            "sindy_wins": wins, "baseline_wins": losses,
            "ties": int((a == piv[b]).sum()),
            "delta_pp": float((a.mean() - piv[b].mean()) * 100),
            "mcnemar_p": mcnemar_exact(wins, losses),
        }
    out["four_wheel"] = {
        "summary": fs,
        "rows": len(fw),
        "env_realizations": fw.scenario_name.nunique(),
        "trajectory_seeds": fw.seed.nunique(),
        "paired": paired,
        "alpha": {
            "mean": float(fw.loc[fw.method == "sindy", "alpha_mean"].mean()),
            "std": float(fw.loc[fw.method == "sindy", "alpha_std"].mean()),
            "response": float(fw.loc[fw.method == "sindy", "alpha_response"].mean()),
        },
        "protocol_note": ("Robot-specific secondary validation with disjoint "
                          "calibration (401-404) and evaluation seeds (501-508)."),
    }

    sindy_main = next(x for x in ms if x["method"] == "sindy")["success"]
    sindy_fw = next(x for x in fs if x["method"] == "sindy")["success"]
    best_baseline_fw = max(x["success"] for x in fs if x["method"] != "sindy")
    out["quality_gates"] = {
        "main_data_unchanged": out["provenance"]["main_unchanged"],
        "main_sindy_ge_85pct": sindy_main >= 0.85,
        "prediction_sindy_beats_both": (
            out["prediction"]["sindy_rmse"] < out["prediction"]["rls_rmse"]
            and out["prediction"]["sindy_rmse"] < out["prediction"]["nominal_rmse"]),
        "four_wheel_sindy_noninferior_or_better": sindy_fw >= best_baseline_fw - 0.02,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("=" * 68)
    print("SIMULATION AUDIT - frozen data, recomputed statistics")
    print("=" * 68)
    print("\n[1] Main benchmark - 300 episodes per method, 6 scenarios")
    print(f"    {'method':<10} {'success':>8} {'95% CI':<18} {'mean terminal err':>18}")
    for x in ms:
        ci = "[%.3f, %.3f]" % (x["ci_low"], x["ci_high"])
        print(f"    {x['method']:<10} {x['success']*100:7.1f}% {ci:<18} {x['terminal_error_mean']:>18.3f}")

    p = out["prediction"]
    print("\n[2] Post-shift prediction benchmark - %d episodes, 5-step-ahead RMSE" % p["n"])
    print(f"    sindy    {p['sindy_rmse']:.4f}")
    print(f"    standard {p['nominal_rmse']:.4f}   (Param-SINDy gains {p['gain_vs_nominal_pct']:.1f}%)")
    print(f"    rls      {p['rls_rmse']:.4f}   (Param-SINDy gains {p['gain_vs_rls_pct']:.1f}%)")

    print("\n[3] Four-wheel secondary - %d paired rows, seeds 501-508, 5 scenarios" % out["four_wheel"]["rows"])
    print(f"    {'method':<10} {'success':>8} {'95% CI':<18}")
    for x in fs:
        ci = "[%.3f, %.3f]" % (x["ci_low"], x["ci_high"])
        print(f"    {x['method']:<10} {x['success']*100:7.1f}% {ci:<18}")
    for b, v in paired.items():
        print(f"    sindy vs {b:<8}: {v['sindy_wins']}W-{v['baseline_wins']}L-"
              f"{v['ties']}T, +{v['delta_pp']:.1f} pp, exact McNemar p = {v['mcnemar_p']:.5f}")

    print("\nQuality gates:")
    ok = True
    for name, passed in out["quality_gates"].items():
        print(f"    [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    if not ok:
        raise SystemExit("AUDIT FAILED - quality gate regression")
    print("AUDIT PASS")


if __name__ == "__main__":
    main()
