import numpy as np
from ..dynamics import nominal_dynamics, wrap_angle

def obstacle_position(scene,t):
    return scene.obstacle.position + (scene.obstacle_velocity*t if scene.dynamic_obstacle else 0.0)

def reference_command(x,goal,obs_pos,p):
    dx,dy=goal[0]-x[0],goal[1]-x[1]; dist=np.hypot(dx,dy)
    desired=np.arctan2(dy,dx); err=wrap_angle(desired-x[2])
    omega=np.clip(1.8*err,-p.omega_max,p.omega_max)
    v=np.clip(p.v_ref*min(1.0,dist/1.2)*(1-0.55*np.clip(abs(err)/1.5,0,1)),p.v_min,p.v_max)
    return np.array([v,omega])

def _candidate_sequences(x,goal,obs_pos,scene,p,last_u,rng):
    center=reference_command(x,goal,obs_pos,p); H=p.horizon
    seqs=[]
    turn=np.array([-0.75,-0.50,-0.25,0.0,0.25,0.50,0.75]); speed=np.array([0.70,0.90,1.08])
    for dw in turn:
        for sv in speed:
            if len(seqs)>=p.n_candidates-8: break
            s=np.tile(center,(H,1)); s[:,0]*=sv; s[:,1]=center[1]+dw*np.linspace(0.3,1.0,H); seqs.append(np.clip(s,[p.v_min,p.omega_min],[p.v_max,p.omega_max]))
    for dv,dw in [(0,0),(0.15,0),(-0.15,0),(0,0.25),(0,-0.25),(0.15,0.25),(-0.15,-0.25),(0.15,-0.25)]:
        s=np.tile(center,(H,1)); s[:,0]+=dv; s[:,1]+=dw; seqs.append(np.clip(s,[p.v_min,p.omega_min],[p.v_max,p.omega_max]))
    return np.asarray(seqs[:p.n_candidates],dtype=float)

def _score_rollout(traj,controls,goal,obs_pos,safe,p):
    # traj: [C,H,5], controls: [C,H,2]
    last=np.concatenate([controls[:,0:1,:],controls[:,:-1,:]],axis=1)
    clearance=np.linalg.norm(traj[:,:,:2]-obs_pos.reshape(1,1,2),axis=2)-safe
    heading_target=np.arctan2(goal[1]-traj[:,:,1],goal[0]-traj[:,:,0])
    heading_err=np.abs(wrap_angle(heading_target-traj[:,:,2]))
    J=p.goal_weight*np.sum(np.linalg.norm(traj[:,:,:2]-goal.reshape(1,1,2),axis=2)**2,axis=1)
    J+=p.heading_weight*np.sum(heading_err,axis=1)
    J+=p.control_weight*np.sum(np.sum(controls**2,axis=2),axis=1)
    J+=p.smooth_weight*np.sum(np.sum((controls-last)**2,axis=2),axis=1)
    neg=np.minimum(clearance,0.0); near=np.maximum(p.clearance_margin-clearance,0.0)
    J+=p.hard_collision_weight*np.sum((neg<0)*(1+clearance**2),axis=1)
    J+=p.obstacle_weight*np.sum((clearance>=0)*near**2,axis=1)
    J-=p.progress_weight*(traj[:,-1,0]-traj[:,0,0])
    J+=p.goal_weight*2.0*np.linalg.norm(traj[:,-1,:2]-goal.reshape(1,2),axis=1)**2
    return J

def _rollout_batch(x, sequences, model_fun, dt):
    C,H,_=sequences.shape
    xx=np.repeat(np.asarray(x,float)[None,:],C,axis=0)
    traj=np.empty((C,H,5),dtype=float)
    for k in range(H):
        xx=model_fun(xx,sequences[:,k,:],dt); traj[:,k,:]=xx
    return traj

def shooting_mpc(x,goal,scene,model_fun,p,dt,last_u,rng,t_now=0.0):
    obs_pos=obstacle_position(scene,t_now); seqs=_candidate_sequences(x,goal,obs_pos,scene,p,last_u,rng)
    # Shared candidate set and vectorized rollout: exactly the same candidates as the reference formulation,
    # but evaluated in one NumPy batch to avoid millions of tiny Python calls.
    traj=_rollout_batch(x,seqs,model_fun,dt)
    scores=_score_rollout(traj,seqs,goal,obs_pos,scene.obstacle.safe_distance,p)
    du=seqs[:,0,:]-last_u.reshape(1,2)
    valid=(np.abs(du[:,0])<=p.max_delta_v)&(np.abs(du[:,1])<=p.max_delta_omega)
    scores=np.where(valid,scores,np.inf)
    idx=int(np.argmin(scores))
    if not np.isfinite(scores[idx]):
        best_seq=np.tile(reference_command(x,goal,obs_pos,p),(p.horizon,1)); best_score=1e18
    else:
        best_seq=seqs[idx]; best_score=float(scores[idx])
    u=best_seq[0].copy(); u[0]=np.clip(u[0],last_u[0]-p.max_delta_v,last_u[0]+p.max_delta_v); u[1]=np.clip(u[1],last_u[1]-p.max_delta_omega,last_u[1]+p.max_delta_omega)
    return u,best_score

def make_model_fun(model,alpha):
    def f(x,u,dt):
        if model is None or alpha<=0: return nominal_dynamics(x,u,dt)
        nom=nominal_dynamics(x,u,dt); learned=model.predict_next(x,u,dt)
        a = model.effective_alpha(x,u,alpha,dt) if hasattr(model,'effective_alpha') else alpha
        if np.ndim(a)>0: a=np.asarray(a)[:,None]
        return (1.0-a)*nom + a*learned
    return f
