# Simulation study

Frozen benchmark data plus the audit / figure pipeline. Python only, no ROS.

## Commands

```bash
python -m pip install -r requirements.txt

python run_simulation.py                      # audit + figures
python run_simulation.py --rerun-four-wheel   # + regenerate four-wheel rows

python audit.py              # statistics + quality gates -> results/audit.json
python make_figures.py       # figures -> results/figures/
python run_four_wheel.py --compare   # re-run locked seeds 501-508 + compare
```

## Data (frozen — never rewritten by any script)

| file | contents |
|---|---|
| `data/main/main.csv` | 900 rows: 300 episodes x 3 methods (sindy / standard / rls) over 6 scenarios, success + diagnostics. SHA-256 checked by `audit.py`. |
| `data/prediction/prediction_benchmark.csv` | 300 rows: post-shift 5-step prediction RMSE (sindy / nominal / rls). |
| `data/four_wheel/fourwheel.csv` | 120 rows: 5 scenarios x 8 seeds x 3 methods on the hidden 9-state wheel/actuator plant, evaluation seeds 501-508 (calibration seeds 401-404 are disjoint). |
| `data/four_wheel/calibration_note.json` | documents the frozen controller calibration used above. |

Method names: `sindy` = Param-SINDy (sparse online identification + adaptive
authority), `standard` = nominal-model controller, `rls` = dense online RLS
identification.

## Results (recomputed by `audit.py`)

- Main benchmark: sindy 99.0% vs standard 85.7% vs rls 85.7%.
- Prediction: sindy RMSE 0.099 m vs nominal 0.432 m (-77.0%) and rls 0.267 m (-62.8%).
- Four-wheel: sindy 95.0% vs 80.0% / 80.0%; paired 6W-0L-34T vs both baselines,
  exact McNemar p = 0.03125; the whole gap is FW3 soft soil.
- Re-running `run_four_wheel.py --compare` reproduces the frozen four-wheel
  rows 120/120.

## Quality gates (exit non-zero on failure)

1. `main_data_unchanged` — frozen main CSV matches its pinned SHA-256.
2. `main_sindy_ge_85pct` — main success >= 85%.
3. `prediction_sindy_beats_both` — prediction RMSE beats both baselines.
4. `four_wheel_sindy_noninferior_or_better` — within 2 pp of the best baseline.
