from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np, pandas as pd
from ..config import set_params
from ..config.scenarios import Scenario, Obstacle
from ..dynamics.base import nominal_dynamics, wrap_angle
from ..control.mpc import shooting_mpc, make_model_fun, obstacle_position
from ..sindy.library import build_feature_matrix
from ..sindy.model import update_model
from ..sindy.rls import init_rls, update_rls

FW_VERSION='fw-protocol-v1'
TRACK=0.56; R=0.115
@dataclass(frozen=True)
class FWCfg:
    tau: float; wtau: float; sl: float; sr: float; lat: float; noise: float; biasx: float; biasy: float; biash: float; shifts: float; post: float

SPECS=[
('FW1 dry grass',.92,.90,.04,.04,.08,.0015,.001,-.001,.001,np.inf,1.0),
('FW2 wet grass',.68,.72,.13,.10,.12,.002,.002,-.0015,.0015,np.inf,1.0),
('FW3 soft soil',.60,.66,.21,.18,.15,.0022,.0025,-.002,.002,np.inf,1.0),
('FW4 asymmetric mud',.58,.64,.28,.18,.16,.0025,.003,-.0025,.003,np.inf,1.0),
('FW5 terrain switch',.82,.84,.10,.09,.13,.002,.002,-.0015,.002,16.0,1.65),
]

def build_fourwheel_scenarios(seed=61010):
    rng=np.random.default_rng(seed); out=[]
    for i,(name,fv,fw,sl,sr,lat,noise,bx,by,bh,shift,post) in enumerate(SPECS):
        ox=[4.0,4.7,4.9,6.2,4.5][i]+rng.uniform(-.1,.1); oy=[2.0,1.8,1.4,4.0,2.3][i]+rng.uniform(-.08,.08)
        gx=[8.2,8.3,8.4,8.8,8.4][i]+rng.uniform(-.05,.05); gy=[2.2,2.0,1.8,2.0,2.8][i]+rng.uniform(-.05,.05)
        asym=i==3 or i==4
        out.append(Scenario(name,i,Obstacle(np.array([ox,oy]),[.48,.46,.44,.42,.46][i]),np.array([gx,gy]),26.0,asym,fv,fw,sl, i==4, np.array([.06,0.]) if i==4 else np.zeros(2), .08+.02*i, .015 if asym else 0., .004 if asym else 0., shift, .55 if i==4 else None, .60 if i==4 else None, .28 if i==4 else None, True))
    return out

CFG={s[0]:FWCfg(0.10+0.01*i,0.08,s[3],s[4],s[5],s[6],s[7],s[8],s[9],s[10],s[11]) for i,s in enumerate(SPECS)}

def meas(x,c,rng):
    y=np.asarray(x[:5],float).copy(); y[0:2]+=np.array([c.biasx,c.biasy])+c.noise*rng.standard_normal(2); y[2]=wrap_angle(y[2]+c.biash+c.noise*.5*rng.standard_normal()); y[3]+=c.noise*rng.standard_normal(); y[4]+=c.noise*rng.standard_normal(); return y

def step(h,u,dt,scene,c,rng):
    x,y,th,v,w,wL,wR,sL,sR=h
    vl=(u[0]-.5*TRACK*u[1])/R; vr=(u[0]+.5*TRACK*u[1])/R
    wL+= (vl-wL)*dt/max(c.tau,dt); wR+=(vr-wR)*dt/max(c.tau,dt)
    targetL=c.sl*(c.post if scene.name=='FW5 terrain switch' and scene.force_full_horizon and h[0]>-999 and False else 1.0)
    # explicit time-based slip below
    tstate=getattr(step,'_t',0.0)
    scale=c.post if tstate>=c.shifts else 1.0
    sl=np.clip(c.sl*scale*(1+.08*np.sin(.25*tstate)),0,.78); sr=np.clip(c.sr*scale*(1+.06*np.cos(.22*tstate)),0,.78)
    if scene.asymmetric: sr*=.93+.05*np.sin(.15*tstate)
    sL+=(sl-sL)*dt/max(c.wtau,dt); sR+=(sr-sR)*dt/max(c.wtau,dt); sL=np.clip(sL,0,.8); sR=np.clip(sR,0,.8)
    vL=R*wL*(1-sL); vR=R*wR*(1-sR); vb=.5*(vL+vR); yaw=(vR-vL)/TRACK + .035*(sL-sR)*vb
    vy=c.lat*(sR-sL)*max(abs(vb),.05)*np.tanh(abs(yaw))
    x+=(vb*np.cos(th)-vy*np.sin(th))*dt; y+=(vb*np.sin(th)+vy*np.cos(th))*dt; th=wrap_angle(th+yaw*dt); v+=(vb-v)*dt/.10; w+=(yaw-w)*dt/.09
    q=np.array([x,y,th,np.clip(v,-1.8,1.8),np.clip(w,-1.5,1.5),wL,wR,sL,sR],float); q[:2]+=0.0004*rng.standard_normal(2); q[2]=wrap_angle(q[2]+.00015*rng.standard_normal()); return q

def one(scene,p,seed,method):
    rng=np.random.default_rng(seed); dt=p.dt; N=int(scene.T/dt); h=np.zeros(9); last=np.zeros(2); model=None; states=[];us=[];nxt=[]; a=[]; reg=[]; res=[]; upd=acc=rej=0
    goal=scene.goal; c=CFG[scene.name]
    for k in range(N):
        t=k*dt; step._t=t; xm=meas(h,c,rng); mf=make_model_fun(model,model.alpha) if method=='sindy' and model else nominal_dynamics
        u,_=shooting_mpc(xm,goal,scene,mf,p.mpc,dt,last,rng,t); hn=step(h,u,dt,scene,c,rng); xnm=meas(hn,c,rng)
        states.append(xm);us.append(u.copy());nxt.append(xnm.copy())
        if method=='sindy' and model is not None:
            try:
                mon=model.update_causal_residual(xm,u,xnm,dt); a.append(float(model.effective_alpha(xm,u,model.alpha,dt))); reg.append(mon['regime_score']); res.append(mon['learned_residual'])
            except Exception: pass
        h=hn; last=u
        if method=='sindy' and k+1>=p.sindy.first_update and len(states)>=p.sindy.min_samples and k+1>=upd+p.sindy.update_freq:
            X=np.asarray(states[-p.sindy.window_size:]);U=np.asarray(us[-p.sindy.window_size:]);Xn=np.asarray(nxt[-p.sindy.window_size:]); mdl,reason=update_model(X,U,Xn,dt,p.sindy,model); upd=k+1
            if mdl is None: rej+=1
            else: model=mdl; acc+=1
        if np.linalg.norm(h[:2]-goal)<=p.mpc.goal_tolerance and k+4<N:
            ok=True
            for j in range(4):
                xm=meas(h,c,rng); mf=make_model_fun(model,model.alpha) if method=='sindy' and model else nominal_dynamics; uh,_=shooting_mpc(xm,goal,scene,mf,p.mpc,dt,last,rng,t+j*dt); h=step(h,uh,dt,scene,c,rng); last=uh
                if np.linalg.norm(h[:2]-goal)>p.mpc.goal_tolerance: ok=False; break
            if ok: break
    arr=np.asarray(states); ctrl=np.asarray(us); md=min((np.linalg.norm(arr[i,:2]-obstacle_position(scene,i*dt)) for i in range(len(arr))),default=np.inf); term=float(np.linalg.norm(h[:2]-goal)); suc=int(term<=p.mpc.goal_tolerance and md>=scene.obstacle.safe_distance)
    return dict(success=suc,terminal_error=term,min_dist=md,control_energy=float(np.sum(ctrl**2)*dt),arrival_time=len(arr)*dt if suc else np.nan,steps=len(arr),alpha_mean=np.nanmean(a) if a else np.nan,alpha_std=np.nanstd(a) if a else np.nan,alpha_response=(np.nanmean(a[-20:])-np.nanmean(a[:20])) if len(a)>=40 else np.nan,regime_mean=np.nanmean(reg) if reg else np.nan,residual_mean=np.nanmean(res) if res else np.nan,accepted=acc,rejected=rej)

def rls_one(scene,p,seed):
    rng=np.random.default_rng(seed); dt=p.dt; N=int(scene.T/dt); h=np.zeros(9);last=np.zeros(2);model=None;X=[];U=[];XN=[];c=CFG[scene.name]
    for k in range(N):
        t=k*dt; step._t=t; xm=meas(h,c,rng); mf=make_model_fun(model,model.alpha) if model else nominal_dynamics; u,_=shooting_mpc(xm,scene.goal,scene,mf,p.mpc,dt,last,rng,t); hn=step(h,u,dt,scene,c,rng); xnm=meas(hn,c,rng);X.append(xm);U.append(u);XN.append(xnm);h=hn;last=u
        if k+1>=15:
            if model is None:
                phi,_=build_feature_matrix(np.asarray(X[-1:]),np.asarray(U[-1:])); model=init_rls(phi.shape[1],alpha=.2,forgetting=.995)
            model=update_rls(model,X[-1],U[-1],XN[-1],dt,forgetting=.995,alpha=.2)
        if np.linalg.norm(h[:2]-scene.goal)<=p.mpc.goal_tolerance: break
    arr=np.asarray(X);ctrl=np.asarray(U);md=min((np.linalg.norm(arr[i,:2]-obstacle_position(scene,i*dt)) for i in range(len(arr))),default=np.inf);term=float(np.linalg.norm(h[:2]-scene.goal)); suc=int(term<=p.mpc.goal_tolerance and md>=scene.obstacle.safe_distance)
    return dict(success=suc,terminal_error=term,min_dist=md,control_energy=float(np.sum(ctrl**2)*dt),arrival_time=len(arr)*dt if suc else np.nan,steps=len(arr),alpha_mean=np.nan,alpha_std=np.nan,alpha_response=np.nan,regime_mean=np.nan,residual_mean=np.nan,accepted=0,rejected=0)

def run(seeds, out_csv):
    p=set_params(); p=replace(p, mpc=replace(p.mpc,horizon=5,n_candidates=16,clearance_margin=.45), sindy=replace(p.sindy,window_size=60,update_freq=5,first_update=15))
    rows=[]; scenes=build_fourwheel_scenarios()
    for ii,seed in enumerate(seeds,1):
        for sid,sc in enumerate(scenes,1):
            for m in ('sindy','standard','rls'):
                r=rls_one(sc,p,seed) if m=='rls' else one(sc,p,seed,m); rows.append({'seed':seed,'scenario_id':sid,'scenario_name':sc.name,'method':m,**r,'protocol_version':FW_VERSION})
        print(f'[fw] {ii}/{len(seeds)} seeds done',flush=True)
    df=pd.DataFrame(rows); Path(out_csv).parent.mkdir(parents=True,exist_ok=True); df.to_csv(out_csv,index=False); return df
