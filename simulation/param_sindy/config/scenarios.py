from dataclasses import dataclass, field
import numpy as np

@dataclass(frozen=True)
class Obstacle:
    position: np.ndarray
    safe_distance: float

@dataclass(frozen=True)
class Scenario:
    name: str
    terrain_type: int
    obstacle: Obstacle
    goal: np.ndarray
    T: float = 45.0
    asymmetric: bool = False
    base_friction_v: float = 1.0
    base_friction_w: float = 1.0
    slip_gain: float = 0.0
    dynamic_obstacle: bool = False
    obstacle_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    terrain_variation: float = 0.0
    heading_bias: float = 0.0
    slip_bias: float = 0.0
    regime_shift_time: float = np.inf
    post_base_friction_v: object = None
    post_base_friction_w: object = None
    post_slip_gain: object = None
    force_full_horizon: bool = False


def define_scenarios():
    return [
        Scenario('S1 Slippery-surface avoidance', 2, Obstacle(np.array([3.5,2.2]),0.50), np.array([8.0,2.0]), 45, False, 0.60,0.70,0.12,False,np.zeros(2),0.02,0.010,0.000),
        Scenario('S2 One-sided mud pit (benchmark)', 1, Obstacle(np.array([5.0,1.10]),0.34), np.array([8.1,1.35]), 45, True,0.72,0.76,0.20,False,np.zeros(2),0.05,0.020,0.005),
        Scenario('S3 One-sided mud pit (stress)', 1, Obstacle(np.array([5.5,1.20]),0.38), np.array([8.9,1.45]), 45, True,0.56,0.58,0.24,False,np.zeros(2),0.08,0.030,0.008),
        Scenario('S4 Emergency avoidance with moving obstacle', 0, Obstacle(np.array([4.5,2.5]),0.38), np.array([8.0,3.5]), 45, False,0.82,0.86,0.10,True,np.array([0.07,0.0]),0.04,0.015,0.004),
        Scenario('S5 Continuously varying terrain', 4, Obstacle(np.array([5.0,2.0]),0.45), np.array([9.0,2.5]), 45, False,0.78,0.80,0.15,False,np.zeros(2),0.15,0.025,0.010),
        Scenario('S6 High-mismatch asymmetric mud stress', 5, Obstacle(np.array([5.5,1.20]),0.38), np.array([8.9,1.45]), 45, True,0.50,0.52,0.30,False,np.zeros(2),0.12,0.050,0.012),
    ]

def define_holdout_scenarios():
    return [
        Scenario('H1 Unseen slippery + asymmetry',2,Obstacle(np.array([3.8,1.9]),0.50),np.array([8.5,2.0]),45,True,0.52,0.48,0.30,False,np.zeros(2),0.18,0.200,0.060),
        Scenario('H2 Unseen continuous terrain',4,Obstacle(np.array([4.2,1.6]),0.48),np.array([9.0,2.0]),45,True,0.48,0.50,0.28,False,np.zeros(2),0.35,0.220,0.080),
        Scenario('H3 Unseen dynamic obstacle',0,Obstacle(np.array([4.2,2.0]),0.42),np.array([8.6,3.0]),45,False,0.65,0.56,0.20,True,np.array([0.14,0.06]),0.12,0.130,0.050),
    ]

def define_generated_holdout_scenarios(n=8, seed=2026):
    """Frozen distribution-defined unseen scenarios; parameters are sampled once and then locked."""
    rng=np.random.default_rng(seed); out=[]
    for i in range(n):
        moving=bool(i%2); asym=bool((i//2)%2)
        bx=float(rng.uniform(0.46,0.70)); bw=float(rng.uniform(0.46,0.72)); slip=float(rng.uniform(0.18,0.34))
        tv=float(rng.uniform(0.18,0.38)); hb=float(rng.uniform(0.08,0.20)); sb=float(rng.uniform(0.04,0.10))
        ox=float(rng.uniform(3.8,5.0)); oy=float(rng.uniform(1.4,2.6)); gx=float(rng.uniform(8.2,9.2)); gy=float(rng.uniform(1.8,3.2))
        ov=np.array([rng.uniform(0.09,0.16), rng.uniform(-0.08,0.08)]) if moving else np.zeros(2)
        out.append(Scenario(f'HG{i+1} frozen unseen draw',i%5,Obstacle(np.array([ox,oy]),float(rng.uniform(0.38,0.50))),np.array([gx,gy]),45,asym,bx,bw,slip,moving,ov,tv,hb,sb))
    return out
