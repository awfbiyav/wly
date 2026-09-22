#!/usr/bin/env python3
"""Render the headline result figures from the frozen CSVs.

Outputs into results/figures/:
    fig1_main_success.png        main benchmark success rate (+95% Wilson CI)
    fig2_prediction_rmse.png     post-shift prediction RMSE per method
    fig3_four_wheel_success.png  four-wheel success per scenario and method
    fig4_four_wheel_terminal.png four-wheel terminal error distribution

Also prints the headline numbers. The frozen CSVs are only read, never written.
"""
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "results" / "figures"
COLORS = {"sindy": "#1b7837", "standard": "#8c510a", "rls": "#2166ac"}
LABELS = {"sindy": "Param-SINDy", "standard": "Nominal", "rls": "Online RLS"}
METHODS = ("sindy", "standard", "rls")


def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hh = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - hh, c + hh


def fig_main_success(main_df):
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=150)
    for i, m in enumerate(METHODS):
        d = main_df[main_df.method == m]
        k = int(d.success.sum())
        n = len(d)
        lo, hi = wilson(k, n)
        ax.bar(i, 100 * k / n, color=COLORS[m], width=0.62,
               label=LABELS[m])
        ax.errorbar(i, 100 * k / n,
                    yerr=[[100 * (k / n - lo)], [100 * (hi - k / n)]],
                    color="black", capsize=5, linewidth=1.2)
        ax.text(i, 102.2, f"{100 * k / n:.1f}%", ha="center", fontsize=10,
                fontweight="bold")
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([LABELS[m] for m in METHODS], fontsize=10)
    ax.set_ylim(0, 112)
    ax.set_ylabel("success rate [%]")
    ax.set_title("Main benchmark: 300 episodes per method\n(6 scenarios, unseen holdout)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_main_success.png")
    plt.close(fig)


def fig_prediction_rmse(pred_df):
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=150)
    names = [("sindy", "sindy_rmse"), ("rls", "rls_rmse"), ("standard", "nominal_rmse")]
    for i, (m, col) in enumerate(names):
        vals = pred_df[col].to_numpy(float)
        mean = vals.mean()
        ci = 1.959963984540054 * vals.std(ddof=1) / math.sqrt(len(vals))
        ax.bar(i, mean, color=COLORS[m], width=0.62, label=LABELS[m])
        ax.errorbar(i, mean, yerr=[[ci], [ci]], color="black", capsize=5, linewidth=1.2)
        ax.text(i, mean + ci + 0.012, f"{mean:.3f}", ha="center", fontsize=10,
                fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels([LABELS[m] for m, _ in names], fontsize=10)
    ax.set_ylabel("post-shift prediction RMSE [m]")
    ax.set_title("Post-shift 5-step prediction\n(300 episodes, lower is better)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_prediction_rmse.png")
    plt.close(fig)


def fig_four_wheel_success(fw_df):
    scenes = sorted(fw_df.scenario_name.unique())
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    width = 0.25
    xs = np.arange(len(scenes))
    for j, m in enumerate(METHODS):
        rates = []
        for sc in scenes:
            d = fw_df[(fw_df.scenario_name == sc) & (fw_df.method == m)]
            rates.append(100 * d.success.mean())
        ax.bar(xs + (j - 1) * width, rates, width=width, color=COLORS[m],
               label=LABELS[m])
        for x, r in zip(xs, rates):
            ax.text(x + (j - 1) * width, r + 1.5, f"{r:.0f}", ha="center",
                    fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([s.replace(" ", "\n", 1) for s in scenes], fontsize=8.5)
    ax.set_ylabel("success rate [%]")
    ax.set_ylim(0, 115)
    ax.set_title("Four-wheel secondary validation: 8 seeds x 5 scenarios per method", fontsize=10)
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_four_wheel_success.png")
    plt.close(fig)


def fig_four_wheel_terminal(fw_df):
    fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=150)
    data = [fw_df.loc[fw_df.method == m, "terminal_error"].to_numpy(float) for m in METHODS]
    bp = ax.boxplot(data, tick_labels=[LABELS[m] for m in METHODS], showfliers=False,
                    patch_artist=True, widths=0.5)
    for patch, m in zip(bp["boxes"], METHODS):
        patch.set_facecolor(COLORS[m])
        patch.set_alpha(0.6)
    ax.set_ylim(0.10, 0.52)
    ax.axhline(0.38, color="crimson", linestyle="--", linewidth=1.2)
    ax.text(2.55, 0.386, "goal tolerance 0.38 m", color="crimson", fontsize=8.5,
            ha="right", va="bottom")
    ax.set_ylabel("terminal error to goal [m]")
    ax.set_title("Four-wheel validation: terminal error\n(seeds 501-508, 5 scenarios)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_four_wheel_terminal.png")
    plt.close(fig)


def fig_alpha_switch():
    """Closed-loop authority collapse at a mid-episode terrain shift.

    Deterministic: the locked dynamic-shift scenario DS1 (shift time is a
    frozen property of the scenario) driven with the Param-SINDy controller,
    seed 501. Shows the mechanism behind the benchmark results: the authority
    weight collapses when the identified model becomes inconsistent with the
    new terrain, then recovers as the model re-adapts.
    """
    from param_sindy.config import set_params
    from param_sindy.simulation import run_scenario
    from param_sindy.study.dynamic_shift import build_dynamic_shift_scenarios

    scene = build_dynamic_shift_scenarios()[0]
    res = run_scenario(scene, set_params(), 501, ("sindy",))["sindy"]
    t = np.asarray(res["effective_alpha_times"], float)
    a = np.asarray(res["effective_alpha_history"], float)
    ok = np.isfinite(t) & np.isfinite(a)
    t, a = t[ok], a[ok]

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    ax.plot(t, a, color=COLORS["sindy"], linewidth=1.6, label="effective authority α")
    ax.axvline(scene.regime_shift_time, color="crimson", linestyle="--", linewidth=1.2,
               label=f"terrain shift (t = {scene.regime_shift_time:.1f} s)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("effective authority α [–]")
    ax.set_ylim(0, max(0.12, a.max() * 1.25))
    ax.set_title("Adaptive authority around a mid-episode terrain shift\n"
                 "(closed loop, dynamic-shift benchmark DS1, seed 501)", fontsize=10)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_alpha_switch.png")
    plt.close(fig)

    pre = a[t < scene.regime_shift_time]
    post = a[(t >= scene.regime_shift_time) & (t < scene.regime_shift_time + 1.5)]
    print(f"authority  pre-shift mean {pre.mean():.3f} -> post-shift mean {post.mean():.3f} "
          f"(collapse {(1 - post.mean() / pre.mean()) * 100:.0f}%)")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    main_df = pd.read_csv(ROOT / "data" / "main" / "main.csv")
    pred_df = pd.read_csv(ROOT / "data" / "prediction" / "prediction_benchmark.csv")
    fw_df = pd.read_csv(ROOT / "data" / "four_wheel" / "fourwheel.csv")

    fig_main_success(main_df)
    fig_prediction_rmse(pred_df)
    fig_four_wheel_success(fw_df)
    fig_four_wheel_terminal(fw_df)
    fig_alpha_switch()

    print("Headline numbers")
    print("----------------")
    for m in METHODS:
        d = main_df[main_df.method == m]
        print(f"main       {LABELS[m]:<12} {100 * d.success.mean():5.1f}%  (n={len(d)})")
    print()
    for m in METHODS:
        d = fw_df[fw_df.method == m]
        print(f"four-wheel {LABELS[m]:<12} {100 * d.success.mean():5.1f}%  (n={len(d)})")
    print()
    sindy = pred_df.sindy_rmse.mean()
    print(f"prediction Param-SINDy {sindy:.4f} m vs Nominal {pred_df.nominal_rmse.mean():.4f} m "
          f"(-{100 * (1 - sindy / pred_df.nominal_rmse.mean()):.1f}%)")
    print(f"prediction Param-SINDy {sindy:.4f} m vs Online RLS {pred_df.rls_rmse.mean():.4f} m "
          f"(-{100 * (1 - sindy / pred_df.rls_rmse.mean()):.1f}%)")
    for f in sorted(FIG.glob("*.png")):
        print(f"wrote {f.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
