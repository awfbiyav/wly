from dataclasses import dataclass, field
import numpy as np

@dataclass(frozen=True)
class MPCParams:
    horizon: int = 5
    dt: float = 0.10
    v_min: float = 0.0
    v_max: float = 1.6
    omega_min: float = -1.2
    omega_max: float = 1.2
    n_candidates: int = 16
    goal_weight: float = 6.0
    heading_weight: float = 1.2
    progress_weight: float = 3.5
    obstacle_weight: float = 260.0
    hard_collision_weight: float = 3.5e4
    control_weight: float = 0.08
    smooth_weight: float = 0.35
    clearance_margin: float = 0.45
    v_ref: float = 1.0
    goal_tolerance: float = 0.38
    max_delta_v: float = 0.30
    max_delta_omega: float = 0.40

@dataclass(frozen=True)
class SINDyParams:
    window_size: int = 60
    update_freq: int = 5
    first_update: int = 15
    min_samples: int = 22
    validation_ratio: float = 0.25
    threshold: float = 0.025
    ridge: float = 1e-5
    stlsq_iters: int = 8
    max_terms_per_channel: int = 5
    min_prediction_gain: float = 0.03
    min_recent_gain: float = 0.02
    max_norm_id_error: float = 0.35
    alpha_min: float = 0.02
    alpha_max: float = 0.75
    confidence_min: float = 0.55
    gain_scale: float = 0.10
    recent_scale: float = 0.08
    uncertainty_scale: float = 0.30
    alpha_smoothing: float = 0.25
    alpha_mode: str = "adaptive"
    fixed_alpha: float = 0.45

@dataclass(frozen=True)
class Params:
    dt: float = 0.10
    T: float = 45.0
    x0: np.ndarray = field(default_factory=lambda: np.zeros(5))
    mpc: MPCParams = field(default_factory=MPCParams)
    sindy: SINDyParams = field(default_factory=SINDyParams)

def set_params():
    return Params()
