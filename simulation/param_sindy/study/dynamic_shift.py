"""Dynamic-shift / out-of-distribution benchmark.

Locked scenario definitions and seeds. Model authority is logged at every
control step (not only at identification updates), and recovery is measured
from the post-shift error peak.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from ..config.scenarios import Scenario, Obstacle
from ..config import set_params
from dataclasses import replace
from ..simulation import run_scenario
from ..utils.metrics import compute_metrics

GEN_SEED = 73021
N_SCENES = 6
SCENARIO_VERSION = "DS-OOD-v5-turn-critical-shift"


def build_dynamic_shift_scenarios(seed: int = GEN_SEED, n: int = N_SCENES):
    """Calibrated regime-shift avoidance benchmark: shift precedes a turn-critical obstacle."""
    rng=np.random.default_rng(seed); out=[]
    for i in range(n):
        pre_v=float(rng.uniform(0.78,0.90)); pre_w=float(rng.uniform(0.82,0.95)); pre_s=float(rng.uniform(0.04,0.08))
        post_v=float(rng.uniform(0.48,0.62)); post_w=float(rng.uniform(0.30,0.45)); post_s=float(rng.uniform(0.20,0.30))
        shift_time=float(rng.uniform(3.4,4.2))
        ox=float(rng.uniform(4.1,4.9)); oy=float(rng.uniform(1.2,2.0)); gx=float(rng.uniform(8.0,8.8)); gy=float(rng.uniform(2.0,3.1))
        safe=float(rng.uniform(0.40,0.46))
        moving=bool(i%2); ov=np.array([rng.uniform(0.03,0.07),rng.uniform(-0.01,0.01)]) if moving else np.zeros(2)
        out.append(Scenario(f"DS{i+1} calibrated shift-before-avoidance", i%5, Obstacle(np.array([ox,oy]),safe), np.array([gx,gy]),
                            25.0, bool(i>=2), pre_v,pre_w,pre_s,moving,ov,float(rng.uniform(0.03,0.08)),0.0,0.0,shift_time,post_v,post_w,post_s,True))
    return out


def serialize_scenarios(scenarios, path: Path, seed: int = GEN_SEED):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows=[]
    for s in scenarios:
        rows.append({'name':s.name,'terrain_type':s.terrain_type,'obstacle_x':float(s.obstacle.position[0]),'obstacle_y':float(s.obstacle.position[1]),
            'safe_distance':float(s.obstacle.safe_distance),'goal_x':float(s.goal[0]),'goal_y':float(s.goal[1]),'T':float(s.T),'asymmetric':bool(s.asymmetric),
            'base_friction_v':float(s.base_friction_v),'base_friction_w':float(s.base_friction_w),'slip_gain':float(s.slip_gain),'dynamic_obstacle':bool(s.dynamic_obstacle),
            'obstacle_vx':float(s.obstacle_velocity[0]),'obstacle_vy':float(s.obstacle_velocity[1]),'terrain_variation':float(s.terrain_variation),
            'heading_bias':float(s.heading_bias),'slip_bias':float(s.slip_bias),'regime_shift_time':float(s.regime_shift_time),
            'post_base_friction_v':float(s.post_base_friction_v),'post_base_friction_w':float(s.post_base_friction_w),'post_slip_gain':float(s.post_slip_gain),'force_full_horizon':bool(getattr(s,'force_full_horizon',False))})
    path.write_text(json.dumps({'version':SCENARIO_VERSION,'seed':seed,'scenarios':rows},indent=2))


def _phase_metrics(result, scene, dt):
    traj=result.get('trajectory'); times=np.arange(len(traj),dtype=float)*dt if traj is not None else np.array([])
    if traj is None or len(traj)==0:
        return {'post_shift_integrated_goal_error':np.nan,'pre_shift_goal_error':np.nan,'post_shift_peak_error':np.nan,'recovery_time':np.nan,'post_shift_steps':0}
    goal=np.asarray(scene.goal,float); d=np.linalg.norm(np.asarray(traj)[:,:2]-goal[None,:],axis=1)
    shift=min(len(d)-1,max(0,int(round(scene.regime_shift_time/dt))))
    pre_start=max(0,shift-int(round(5.0/dt))); pre=d[pre_start:shift] if shift>pre_start else d[:min(len(d),max(1,int(round(5.0/dt))))]
    post=d[shift:]; pre_mean=float(np.mean(pre)) if len(pre) else np.nan
    post_i=float(np.mean(post)) if len(post) else np.nan; peak_idx=int(np.argmax(post)) if len(post) else 0; peak=float(post[peak_idx]) if len(post) else np.nan
    recovery=np.nan
    if len(post) and np.isfinite(pre_mean):
        target=pre_mean*1.05
        after=post[peak_idx:]
        # recovery is time from the post-shift error peak until the trajectory
        # re-enters a 5% band around the pre-shift operating error.
        idx=np.flatnonzero(after<=target)
        recovery=float(idx[0]*dt) if len(idx) else float(len(after)*dt)
    return {'post_shift_integrated_goal_error':post_i,'pre_shift_goal_error':pre_mean,'post_shift_peak_error':peak,
            'recovery_time':recovery,'post_shift_steps':int(len(post))}


def _alpha_phase_metrics(result, scene):
    times=np.asarray(result.get('effective_alpha_times',[]),float); alphas=np.asarray(result.get('effective_alpha_history',[]),float)
    good=np.isfinite(alphas)&np.isfinite(times)
    times=times[good]; alphas=alphas[good]
    if len(alphas)==0: return {'pre_shift_alpha':np.nan,'post_shift_alpha':np.nan,'alpha_response':np.nan,'alpha_std':np.nan,'alpha_abs_response':np.nan}
    pre=alphas[times<scene.regime_shift_time]; post=alphas[times>=scene.regime_shift_time]
    pre_mean=float(np.mean(pre)) if len(pre) else np.nan; post_mean=float(np.mean(post)) if len(post) else np.nan
    return {'pre_shift_alpha':pre_mean,'post_shift_alpha':post_mean,'alpha_response':float(post_mean-pre_mean) if len(pre) and len(post) else np.nan,
            'alpha_abs_response':float(abs(post_mean-pre_mean)) if len(pre) and len(post) else np.nan,'alpha_std':float(np.std(alphas)) if len(alphas) else np.nan}


def _quality_phase_metrics(result, scene):
    t=np.asarray(result.get('quality_times',[]),float); pg=np.asarray(result.get('pred_gain_history',[]),float); rg=np.asarray(result.get('recent_gain_history',[]),float)
    cf=np.asarray(result.get('confidence_history',[]),float); ni=np.asarray(result.get('norm_id_history',[]),float); nn=np.asarray(result.get('nnz_history',[]),float)
    def phase(v, after=False):
        if len(v)!=len(t) or len(v)==0: return np.nan
        mask=t>=scene.regime_shift_time if after else t<scene.regime_shift_time
        return float(np.mean(v[mask])) if np.any(mask) else np.nan
    return {'pre_shift_pred_gain':phase(pg,False),'post_shift_pred_gain':phase(pg,True),'pre_shift_confidence':phase(cf,False),'post_shift_confidence':phase(cf,True),
            'pre_shift_nnz':phase(nn,False),'post_shift_nnz':phase(nn,True),'post_shift_norm_id_error':phase(ni,True)}




def _regime_metrics(result, scene):
    t=np.asarray(result.get('residual_times',[]),float); rr=np.asarray(result.get('residual_ratio_history',[]),float); rs=np.asarray(result.get('regime_score_history',[]),float); ra=np.asarray(result.get('regime_alarm_history',[]),float)
    lr=np.asarray(result.get('learned_resid_history',[]),float); nr=np.asarray(result.get('nominal_resid_history',[]),float)
    if len(t)==0:
        return {'pre_shift_residual_ratio':np.nan,'post_shift_residual_ratio':np.nan,'post_shift_regime_score':np.nan,'regime_alarm_rate':np.nan,'regime_response':np.nan,'post_shift_learned_pred_error':np.nan,'post_shift_nominal_pred_error':np.nan,'prediction_improvement_pct':np.nan}
    pre=t<scene.regime_shift_time; post=~pre
    pre_rr=float(np.mean(rr[pre])) if np.any(pre) else np.nan; post_rr=float(np.mean(rr[post])) if np.any(post) else np.nan
    pre_rs=float(np.mean(rs[pre])) if np.any(pre) else 0.0; post_rs=float(np.mean(rs[post])) if np.any(post) else np.nan
    alarm=float(np.mean(ra[post])) if np.any(post) else np.nan
    pl=float(np.mean(lr[post])) if np.any(post) else np.nan; pn=float(np.mean(nr[post])) if np.any(post) else np.nan
    imp=float(1.0-pl/(pn+1e-12)) if np.isfinite(pl) and np.isfinite(pn) else np.nan
    return {'pre_shift_residual_ratio':pre_rr,'post_shift_residual_ratio':post_rr,'post_shift_regime_score':post_rs,'regime_alarm_rate':alarm,'regime_response':float(post_rs-pre_rs) if np.isfinite(post_rs) else np.nan,'post_shift_learned_pred_error':pl,'post_shift_nominal_pred_error':pn,'prediction_improvement_pct':100*imp if np.isfinite(imp) else np.nan}

def _seed_rows(seed, methods):
    p=set_params(); p=replace(p, mpc=replace(p.mpc, horizon=5, n_candidates=8), sindy=replace(p.sindy, window_size=45, update_freq=5, first_update=15)); rows=[]
    for scene_id,scene in enumerate(build_dynamic_shift_scenarios(),start=1):
        res=run_scenario(scene,p,int(seed),methods)
        for method,result in res.items():
            m=compute_metrics(result); m.update(_phase_metrics(result,scene,p.dt)); m.update(_alpha_phase_metrics(result,scene)); m.update(_quality_phase_metrics(result,scene)); m.update(_regime_metrics(result,scene))
            m.update({'seed':int(seed),'scenario_id':scene_id,'scenario_name':scene.name,'method':method,'protocol_version':SCENARIO_VERSION})
            rows.append(m)
    return rows


def run_dynamic_shift(seeds,methods=('sindy','standard','rls'),out_csv=None,scenario_json=None,workers=1):
    seeds=list(seeds); rows=[]
    if scenario_json is not None: serialize_scenarios(build_dynamic_shift_scenarios(),Path(scenario_json))
    if workers>1 and len(seeds)>1:
        with ThreadPoolExecutor(max_workers=int(workers)) as ex:
            fs=[ex.submit(_seed_rows,int(seed),tuple(methods)) for seed in seeds]
            for i,f in enumerate(fs,1): rows.extend(f.result()); print(f'[dyn] {i}/{len(fs)} seeds done',flush=True)
    else:
        for i,seed in enumerate(seeds,1): rows.extend(_seed_rows(int(seed),tuple(methods))); print(f'[dyn] {i}/{len(seeds)} seeds done',flush=True)
    df=pd.DataFrame(rows)
    if out_csv: Path(out_csv).parent.mkdir(parents=True,exist_ok=True); df.to_csv(out_csv,index=False)
    return df
