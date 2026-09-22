# sindy+mpc — Terrain-Adaptive MPC with Sparse Online Dynamics Identification

Terrain-robust model predictive control (MPC) for wheeled robots on
low-traction ground.

The controller identifies a **sparse dynamics model online** (SINDy sparse
regression on a sliding window of drive data) and blends it with a nominal
kinematic model using an **adaptive authority weight** `alpha`. A causal
residual monitor reduces the authority whenever the learned model becomes
inconsistent with the current terrain regime — so the MPC exploits the learned
model when it helps and falls back to the safe nominal model when it does not.
Commands are computed by a sampling-based shooting MPC over the blended model.

The repository has two layers:

1. **`simulation/`** — the frozen simulation study (main benchmark, prediction
   benchmark, four-wheel secondary validation) plus the audit and figure
   pipeline. Runs anywhere with Python 3.11+, no ROS required.
2. **`hardware/`** — the ROS 2 Humble package that runs the **same frozen
   code** on a four-wheel robot, with a digital-twin verification chain that
   runs without ROS or robot hardware.

## Results at a glance

All numbers below are recomputed from the frozen data by `simulation/audit.py`
and were re-verified on this release.

**Main benchmark** — 300 episodes per method across 6 terrain scenarios with
a hidden slip/mismatch plant:

| method      | success rate | 95% CI        |
|-------------|-------------:|---------------|
| Param-SINDy MPC | **99.0%**    | [97.1, 99.7]  |
| Nominal MPC | 85.7%        | [81.2, 89.2]  |
| Online RLS MPC | 85.7%        | [81.2, 89.2]  |

**Post-shift prediction** (5-step RMSE after a terrain change, 300 episodes):
Param-SINDy **0.099 m** vs Nominal 0.432 m (−77.0%) and Online RLS 0.267 m
(−62.8%).

**Four-wheel secondary validation** — a hidden 9-state wheel/actuator plant;
the controller sees only the 5-state estimate. 120 paired rows (5 scenarios x
8 seeds x 3 methods), disjoint calibration (401–404) and evaluation
(501–508) seeds:

| method      | success rate | paired vs Param-SINDy MPC        |
|-------------|-------------:|----------------------------------|
| Param-SINDy MPC | **95.0%**    | —                                |
| Nominal MPC | 80.0%        | 6W–0L–34T, +15.0 pp, McNemar p = 0.031 |
| Online RLS MPC | 80.0%        | 6W–0L–34T, +15.0 pp, McNemar p = 0.031 |

The whole advantage is concentrated in the hard scenario (FW3 soft soil),
where the baselines achieve 0% and Param-SINDy MPC 75%. Re-running the
evaluation with `run_four_wheel.py --compare` reproduces the frozen rows
exactly (120/120 row agreement).

**Digital-twin hardware verification** — the ROS-free hardware code path
reproduces the locked simulation behaviour with 96.7% row agreement (gate:
>= 80%). The 2x2 ablation twin attributes the gain: sparse model structure
carries most of the closed-loop improvement, while adaptive authority rescues
dense (RLS) adaptation on soft soil (12.5% -> 50.0%).

## Repository layout

```
sindy+mpc/
├── README.md                         this file
├── DATA_AVAILABILITY.md              what data ships here (simulation/twin only)
├── LICENSE                           Apache-2.0
├── simulation/                       frozen simulation study (Python only)
│   ├── run_simulation.py             one-command pipeline: audit + figures
│   ├── audit.py                      recompute statistics + quality gates
│   ├── run_four_wheel.py             re-run four-wheel evaluation (locked seeds)
│   ├── make_figures.py               render the headline figures
│   ├── param_sindy/                  research core (identification + MPC)
│   ├── data/                         frozen simulation CSVs (never rewritten)
│   └── results/                      audit JSON, figures, re-run outputs
└── hardware/
    └── ros2_ws/src/terrain_adaptive_control/   ROS 2 Humble package
        ├── scripts/verify_all.py              10-gate verification (no ROS)
        ├── terrain_adaptive_control/          adapter nodes
        ├── param_sindy/                       frozen core (hash-manifested)
        ├── docs/                              setup + experiment protocol
        └── data/
            ├── frozen_simulation/             frozen simulation reference
            └── twin/                          digital-twin evidence (synthetic)
```

## Data provenance: simulation vs real hardware

| Layer | In this repo? | What it is |
|---|---|---|
| `simulation/` benchmarks | **yes** | Closed-loop episodes of the simulated plants (deterministic, locked seeds) |
| Digital twin (`hardware/.../data/twin/`) | **yes** | The real adapter code driven against the simulated plant — verifies the software path, not the physics; **synthetic, no physical robot involved** |
| Real hardware | **no** | Field-day logs stay on the operator's machine (`~/terrain_results`) and are shared as hashed zip bundles via `scripts/pack_field_results.py` |

See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for the complete data
inventory and the no-real-robot-data policy.

## Quick start (simulation, no ROS)

```bash
cd simulation
python -m pip install -r requirements.txt
python run_simulation.py                  # audit + figures (~1 min)
python run_simulation.py --rerun-four-wheel   # + regenerate & compare (~1 min)
```

Outputs:

- `results/audit.json` — recomputed statistics and quality gates
- `results/figures/fig1_main_success.png` — main benchmark success rates
- `results/figures/fig2_prediction_rmse.png` — post-shift prediction RMSE
- `results/figures/fig3_four_wheel_success.png` — four-wheel success by scenario
- `results/figures/fig4_four_wheel_terminal.png` — terminal error distributions
- `results/figures/fig5_alpha_switch.png` — closed-loop authority collapse and
  recovery around a mid-episode terrain shift (the mechanism behind the gains)

`audit.py` exits non-zero if any pre-registered quality gate fails, so it can
be used as a CI check.

## Hardware verification (no ROS, no robot)

```bash
cd hardware/ros2_ws/src/terrain_adaptive_control
python -m pip install -r requirements.txt
python scripts/verify_all.py              # 10 gates, ~1 min
```

All gates PASS = the adapter calls the frozen identification / authority /
MPC code with the frozen schedule and reproduces the locked closed-loop
behaviour through the hardware code path. This is the pre-field gate; it does
not claim any physical-robot result.

## Robot bring-up (Ubuntu 22.04 + ROS 2 Humble)

See `hardware/ros2_ws/src/terrain_adaptive_control/docs/HARDWARE_SETUP.md`
for workspace build and safe-start instructions, and
`docs/EXPERIMENT_PROTOCOL.md` for the pre-registered field experiment
protocol (H1 open-loop prediction, H2 closed-loop goal reaching, H3
terrain-switch authority trace, H4 2x2 ablation).

## The research core (`param_sindy`)

| module | contents |
|---|---|
| `sindy/library.py`    | 25-feature library for the 5-state planar model |
| `sindy/model.py`      | STLSQ identification, quality gates, adaptive authority `effective_alpha`, causal regime monitor |
| `sindy/rls.py`        | recursive least-squares baseline identification |
| `control/mpc.py`      | sampling-based shooting MPC (candidate generation, rollout, cost) |
| `dynamics/`           | nominal model, hidden true plant |
| `simulation/scenario.py` | episode loop with online identification |
| `study/`              | benchmark generators (four-wheel validation, prediction benchmark, dynamic shift, RLS baseline) |

The hardware package ships a byte-identical copy of this core guarded by a
SHA-256 manifest (`docs/FROZEN_CORE.sha256`, checked by
`scripts/check_invariants.py`).

## License

Apache-2.0 — see the [LICENSE](LICENSE) file (replace the copyright holder
line with your name or GitHub handle before publishing).
