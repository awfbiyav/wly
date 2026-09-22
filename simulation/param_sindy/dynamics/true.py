import numpy as np
from .base import wrap_angle

def scene_truth(scene,t,rng):
    # Reproducible mismatch sampled per rollout; fixed parameters are hidden from the controller.
    phase = 0.7*np.sin(0.7*t+0.4*rng.random())
    base_v = scene.base_friction_v
    base_w = scene.base_friction_w
    base_s = scene.slip_gain
    if t >= getattr(scene,'regime_shift_time',np.inf):
        if getattr(scene,'post_base_friction_v',None) is not None: base_v=scene.post_base_friction_v
        if getattr(scene,'post_base_friction_w',None) is not None: base_w=scene.post_base_friction_w
        if getattr(scene,'post_slip_gain',None) is not None: base_s=scene.post_slip_gain
    fv = base_v * (1.0 + scene.terrain_variation*0.45*np.sin(0.35*t+phase))
    fw = base_w * (1.0 + scene.terrain_variation*0.35*np.cos(0.28*t+phase))
    fv = float(np.clip(fv,0.40,1.10)); fw=float(np.clip(fw,0.40,1.15))
    slip = base_s*(1.0+0.15*np.sin(0.3*t))
    return fv,fw,slip

def true_step(x,u,dt,scene,t,rng):
    fv,fw,slip_gain=scene_truth(scene,t,rng)
    heading_bias=scene.heading_bias
    slip_bias=scene.slip_bias
    xn = __import__('param_sindy.dynamics.base',fromlist=['base_dynamics']).base_dynamics(x,u,dt,fv,fw,slip_gain)
    xn[2] = wrap_angle(xn[2] + heading_bias*dt)
    xn[0] += -np.sin(x[2])*slip_bias*dt
    xn[1] += np.cos(x[2])*slip_bias*dt
    if scene.dynamic_obstacle:
        # Dynamics are unchanged; obstacle motion is handled in the controller/metric layer.
        pass
    noise=np.array([0.0005,0.0005,0.0002,0.0015,0.0015])*rng.standard_normal(5)
    return xn+noise
