import numpy as np
from ..dynamics import true_step, nominal_dynamics
from ..sindy import update_model
from ..control import shooting_mpc, make_model_fun
from ..control.mpc import obstacle_position


def simulate_one(scene,params,method='sindy',seed=1):
    rng=np.random.default_rng(seed)
    dt=params.dt; N=int(scene.T/dt)
    x=params.x0.copy(); goal=scene.goal; last_u=np.zeros(2)
    states=[]; inputs=[]; next_states=[]
    model=None; updates=0; accepted=0; rejected=0; alpha_hist=[]; alpha_times=[]; gain_hist=[]
    # Control-time traces are distinct from update-time traces.  They capture
    # the actually applied model authority after OOD/disagreement attenuation.
    eff_alpha_hist=[]; eff_alpha_times=[]
    pred_gain_hist=[]; recent_gain_hist=[]; confidence_hist=[]; norm_id_hist=[]; nnz_hist=[]; quality_times=[]
    residual_ratio_hist=[]; regime_score_hist=[]; regime_alarm_hist=[]; residual_times=[]; learned_resid_hist=[]; nominal_resid_hist=[]
    next_update=params.sindy.first_update
    for k in range(N):
        t=k*dt
        if method=='sindy' and model is not None:
            model_fun=make_model_fun(model,model.alpha)
        else:
            model_fun=nominal_dynamics
        u,_=shooting_mpc(x,goal,scene,model_fun,params.mpc,dt,last_u,rng,t)
        if method=='sindy' and model is not None:
            try:
                eff_a=float(model.effective_alpha(x,u,model.alpha,dt))
            except Exception:
                eff_a=float(model.alpha)
            eff_alpha_hist.append(eff_a); eff_alpha_times.append(float(t))
        else:
            eff_alpha_hist.append(np.nan); eff_alpha_times.append(float(t))
        xn=true_step(x,u,dt,scene,t,rng)
        if method=='sindy' and model is not None:
            try:
                rm=model.update_causal_residual(x,u,xn,dt)
                residual_ratio_hist.append(float(rm['ratio'])); regime_score_hist.append(float(rm['regime_score'])); regime_alarm_hist.append(float(rm['regime_alarm'])); learned_resid_hist.append(float(rm['learned_residual'])); nominal_resid_hist.append(float(rm['nominal_residual'])); residual_times.append(float(t))
            except Exception:
                pass
        states.append(x.copy()); inputs.append(u.copy()); next_states.append(xn.copy())
        x=xn; last_u=u
        if method=='sindy' and k+1>=next_update:
            X=np.asarray(states[-params.sindy.window_size:]); U=np.asarray(inputs[-params.sindy.window_size:]); Xn=np.asarray(next_states[-params.sindy.window_size:])
            mdl,reason=update_model(X,U,Xn,dt,params.sindy,model)
            updates+=1
            if mdl is not None:
                model=mdl; accepted+=1; alpha_hist.append(model.alpha); alpha_times.append(float(t)); gain_hist.append((model.pred_gain,model.recent_gain))
                pred_gain_hist.append(float(model.pred_gain)); recent_gain_hist.append(float(model.recent_gain)); confidence_hist.append(float(model.confidence)); norm_id_hist.append(float(model.norm_id_error)); nnz_hist.append(int(model.n_nonzero)); quality_times.append(float(t))
            else:
                rejected+=1
            next_update += params.sindy.update_freq
        pos_obs=obstacle_position(scene,t)
        if (not getattr(scene,'force_full_horizon',False)) and np.linalg.norm(x[:2]-goal) <= params.mpc.goal_tolerance:
            # Require a brief hold for repeatability.
            if k+2 < N:
                hold_ok=True; xh=x.copy(); uh=last_u.copy()
                for j in range(3):
                    model_fun=make_model_fun(model,model.alpha) if method=='sindy' and model else nominal_dynamics
                    uh,_=shooting_mpc(xh,goal,scene,model_fun,params.mpc,dt,uh,rng,t+j*dt)
                    xh=true_step(xh,uh,dt,scene,t+j*dt,rng)
                    if np.linalg.norm(xh[:2]-goal)>params.mpc.goal_tolerance: hold_ok=False
                if hold_ok: x=xh; break
    arr=np.asarray(states); ctrls=np.asarray(inputs)
    if len(arr):
        d=[]
        for i in range(len(arr)):
            d.append(np.linalg.norm(arr[i,:2]-obstacle_position(scene,i*dt)))
        min_dist=float(np.min(d))
    else: min_dist=np.inf
    terminal=float(np.linalg.norm(x[:2]-goal))
    success=bool(terminal<=params.mpc.goal_tolerance and min_dist>=scene.obstacle.safe_distance)
    return {
        'success':success,'terminal_error':terminal,'min_obstacle_distance':min_dist,
        'control_energy':float(np.sum(ctrls**2)*dt) if len(ctrls) else 0.0,
        'trajectory':arr,'controls':ctrls,'update_count':updates,'accepted_update_count':accepted,
        'rejected_update_count':rejected,'model':model.__dict__ if model else None,
        'alpha_history':alpha_hist,'alpha_times':alpha_times,'gain_history':gain_hist,'steps':len(arr),
        'effective_alpha_history':eff_alpha_hist,'effective_alpha_times':eff_alpha_times,
        'pred_gain_history':pred_gain_hist,'recent_gain_history':recent_gain_hist,
        'confidence_history':confidence_hist,'norm_id_history':norm_id_hist,'nnz_history':nnz_hist,'quality_times':quality_times,
        'residual_ratio_history':residual_ratio_hist,'regime_score_history':regime_score_hist,'regime_alarm_history':regime_alarm_hist,'residual_times':residual_times,
        'learned_resid_history':learned_resid_hist,'nominal_resid_history':nominal_resid_hist,
        'arrival_time_success':len(arr)*dt if success else np.nan,
    }


def run_scenario(scene,params,seed=1,methods=('sindy','standard')):
    out={}
    for m in methods:
        if m=='rls':
            from ..study.adaptive_baselines import simulate_rls
            out[m]=simulate_rls(scene,params,seed=seed)
        else: out[m]=simulate_one(scene,params,seed=seed,method=m)
    return out
