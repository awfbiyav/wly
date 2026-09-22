from dataclasses import replace
import numpy as np
from ..sindy.library import build_feature_matrix
from ..sindy.rls import init_rls, update_rls, RLSModel
from ..dynamics import true_step, nominal_dynamics
from ..control import shooting_mpc, make_model_fun
from ..config import define_scenarios
from ..control.mpc import obstacle_position
from ..utils.metrics import compute_metrics

def simulate_rls(scene, params, seed=1, alpha=0.20, forgetting=0.995):
    rng=np.random.default_rng(seed); dt=params.dt; N=int(scene.T/dt)
    x=params.x0.copy(); goal=scene.goal; last_u=np.zeros(2)
    states=[]; inputs=[]; next_states=[]; model=None; accepted=0; update_count=0
    rng_phi=np.zeros((1,1))
    first=15; freq=10
    for k in range(N):
        model_fun=make_model_fun(model,model.alpha) if model is not None else nominal_dynamics
        u,_=shooting_mpc(x,goal,scene,model_fun,params.mpc,dt,last_u,rng,k*dt)
        xn=true_step(x,u,dt,scene,k*dt,rng)
        states.append(x.copy()); inputs.append(u.copy()); next_states.append(xn.copy())
        x=xn; last_u=u
        if k+1>=first:
            if model is None:
                phi,_=build_feature_matrix(np.asarray(states)[-1:,:],np.asarray(inputs)[-1:,:])
                model=init_rls(phi.shape[1],alpha=alpha,forgetting=forgetting)
            # update every step after warm-up for fair online adaptation
            model=update_rls(model,states[-1],inputs[-1],next_states[-1],dt,forgetting=forgetting,alpha=alpha)
            update_count+=1; accepted+=1
        if np.linalg.norm(x[:2]-goal)<=params.mpc.goal_tolerance and k+2<N:
            hold=True; xh=x.copy(); uh=last_u.copy()
            for j in range(3):
                mf=make_model_fun(model,model.alpha) if model else nominal_dynamics
                uh,_=shooting_mpc(xh,goal,scene,mf,params.mpc,dt,uh,rng,k*dt+j*dt)
                xh=true_step(xh,uh,dt,scene,k*dt+j*dt,rng)
                if np.linalg.norm(xh[:2]-goal)>params.mpc.goal_tolerance: hold=False
            if hold: x=xh; break
    arr=np.asarray(states); ctrls=np.asarray(inputs)
    if len(arr):
        md=min(np.linalg.norm(arr[i,:2]-obstacle_position(scene,i*dt))-0 for i in range(len(arr)))
    else: md=np.inf
    terminal=float(np.linalg.norm(x[:2]-goal)); success=bool(terminal<=params.mpc.goal_tolerance and md>=scene.obstacle.safe_distance)
    return {'success':success,'terminal_error':terminal,'min_obstacle_distance':float(md),'control_energy':float(np.sum(ctrls**2)*dt),'update_count':update_count,'accepted_update_count':accepted,'rejected_update_count':0,'model':model.__dict__ if model else None,'alpha_history':[alpha] if model else [],'gain_history':[], 'steps':len(arr),'arrival_time_success':len(arr)*dt if success else np.nan,'trajectory':arr,'controls':ctrls}
