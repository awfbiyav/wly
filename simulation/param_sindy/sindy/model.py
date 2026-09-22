from dataclasses import dataclass
import numpy as np
from .library import build_feature_matrix
from ..dynamics.base import nominal_dynamics, wrap_angle

@dataclass
class SINDyModel:
    Xi: np.ndarray; feature_names: list; confidence: float; alpha: float
    pred_gain: float; recent_gain: float; norm_id_error: float; n_nonzero: int
    train_size: int; version: int=0; mode: str='sindy'; feature_center: np.ndarray=None; feature_scale: np.ndarray=None; ood_scale: float=3.0
    learned_resid_ema: float = np.nan; nominal_resid_ema: float = np.nan;
    regime_score: float = 0.0; regime_alarm: float = 0.0; alpha_reset: float = 0.12;
    learned_ref_ema: float = np.nan; regime_count: int = 0
    local_validity_ema: float = 1.0; local_advantage_ema: float = np.nan
    def effective_alpha(self, x, u, base_alpha, dt=0.10):
        xa=np.asarray(x,float); ua=np.asarray(u,float)
        if self.feature_center is None or self.feature_scale is None:
            raw=np.asarray(base_alpha,float)
            return float(raw) if xa.ndim==1 else np.full(len(xa),float(raw))
        phi,_=build_feature_matrix(xa if xa.ndim==2 else xa[None,:], ua if ua.ndim==2 else ua[None,:])
        z=np.abs((phi-self.feature_center)/np.maximum(self.feature_scale,1e-6))
        dist=np.sqrt(np.mean(np.minimum(z,self.ood_scale)**2,axis=1))
        ood_penalty=1.0/(1.0+0.95*np.maximum(0.0,dist-1.0)**2)
        nom=nominal_dynamics(xa,ua,dt); learn=self.predict_next(xa,ua,dt)
        disagreement=np.linalg.norm(learn[...,:2]-nom[...,:2],axis=-1) if xa.ndim==2 else float(np.linalg.norm(learn[:2]-nom[:2]))
        disagree_penalty=1.0/(1.0+22.0*np.maximum(0.0,disagreement-0.012)**2)
        rs=float(np.clip(self.regime_score,0.0,4.0))
        # A regime alarm is treated as a temporary loss of model authority.
        # The authority recovers only after the recent-fit monitor becomes stable.
        regime_penalty=np.exp(-0.9*rs)
        # Scope-aware local evidence gate: if the learned predictor is currently
        # worse than the nominal predictor on causal residuals, reduce learned
        # authority even when the historical validation gain was positive.
        lv=float(np.clip(self.local_validity_ema,0.0,1.0))
        # Keep the nominal secondary benchmark behavior unchanged unless the
        # causal regime monitor actually raises an alarm. This avoids making
        # sensor-noise robustness depend on a weak residual fluctuation, while
        # still enforcing a safe authority reduction once the learned model
        # is demonstrably inconsistent with the current regime.
        evidence_penalty = (0.25 + 0.75*lv) if self.regime_alarm>0.5 else 1.0
        raw=float(base_alpha)*ood_penalty*disagree_penalty*regime_penalty*evidence_penalty
        out=np.clip(raw,0.0,float(base_alpha))
        if self.regime_alarm>0.5:
            out=np.minimum(out, float(self.alpha_reset))
        if xa.ndim==1:
            # In the scalar-query path all intermediate penalties are 0-D arrays
            # under NumPy broadcasting; normalize explicitly to a Python float.
            return float(np.asarray(out).reshape(()))
        return np.asarray(out, dtype=float).reshape(-1)

    def update_causal_residual(self, x, u, x_next, dt=0.10, ema=0.18):
        """Causal local-regime monitor. Compares current learned error to a slow pre-shift reference, not to nominal alone."""
        nom=nominal_dynamics(np.asarray(x,float),np.asarray(u,float),dt); learn=self.predict_next(x,u,dt)
        xn=np.asarray(x_next,float)
        en=float(np.linalg.norm(nom[:2]-xn[:2]) + 0.25*abs(wrap_angle(nom[2]-xn[2])))
        el=float(np.linalg.norm(learn[:2]-xn[:2]) + 0.25*abs(wrap_angle(learn[2]-xn[2])))
        if not np.isfinite(self.learned_resid_ema):
            self.learned_resid_ema=el; self.nominal_resid_ema=en; self.learned_ref_ema=max(el,1e-4)
        else:
            self.learned_resid_ema=(1-ema)*self.learned_resid_ema+ema*el
            self.nominal_resid_ema=(1-ema)*self.nominal_resid_ema+ema*en
            if not np.isfinite(self.learned_ref_ema):
                self.learned_ref_ema=max(el,1e-4)
        # The reference adapts slowly so a sudden jump in learned residual produces a short alarm.
        ref=self.learned_ref_ema
        ratio=el/(ref+1e-8)
        ref_gain=0.025 if self.regime_alarm<=0.5 else 0.005
        self.learned_ref_ema=(1-ref_gain)*ref+ref_gain*el
        excess=max(0.0,ratio-1.18)
        self.regime_score=float(0.90*self.regime_score+0.10*np.clip(excess*2.8,0.0,4.0))
        # Local predictive advantage relative to nominal. Values below one mean
        # learned prediction is currently better; values above one mean it is worse.
        local_ratio=float(el/(en+1e-8))
        adv=float(np.clip(1.0-local_ratio, -2.0, 1.0))
        target_validity=float(np.clip(0.5+0.5*adv,0.0,1.0))
        self.local_validity_ema=float(0.88*self.local_validity_ema+0.12*target_validity)
        self.local_advantage_ema=float(0.88*(self.local_advantage_ema if np.isfinite(self.local_advantage_ema) else target_validity)+0.12*target_validity)
        alarm_cond=(ratio>1.40 and self.regime_score>0.35)
        if alarm_cond:
            self.regime_alarm=1.0; self.regime_count += 1
        elif self.regime_alarm>0.0:
            self.regime_alarm=max(0.0,self.regime_alarm-0.015)
        return {'learned_residual':el,'nominal_residual':en,'ratio':ratio,'regime_score':self.regime_score,'regime_alarm':self.regime_alarm}

    def predict_channels(self,x,u):
        xa=np.asarray(x,float); ua=np.asarray(u,float)
        if xa.ndim==2:
            phi,_=build_feature_matrix(xa,ua); return phi@self.Xi
        phi,_=build_feature_matrix(xa[None,:],ua[None,:]); return (phi@self.Xi).reshape(-1)
    def predict_next(self,x,u,dt):
        x=np.asarray(x,float); u=np.asarray(u,float); p=self.predict_channels(x,u)
        if x.ndim==2:
            th=x[:,2]; q=x.copy(); q[:,3]=p[:,0]; q[:,4]=p[:,1]
            q[:,0]=x[:,0]+(p[:,0]*np.cos(th)-p[:,2]*np.sin(th))*dt
            q[:,1]=x[:,1]+(p[:,0]*np.sin(th)+p[:,2]*np.cos(th))*dt
            q[:,2]=wrap_angle(th+p[:,1]*dt); return q
        th=x[2]; q=x.copy(); q[3]=p[0]; q[4]=p[1]; q[0]=x[0]+(p[0]*np.cos(th)-p[2]*np.sin(th))*dt; q[1]=x[1]+(p[0]*np.sin(th)+p[2]*np.cos(th))*dt; q[2]=wrap_angle(th+p[1]*dt); return q

def _ridge(Phi,y,ridge): return np.linalg.solve(Phi.T@Phi+ridge*np.eye(Phi.shape[1]),Phi.T@y)
def stlsq(Phi,Y,threshold,ridge,n_iter,max_terms):
    Xi=_ridge(Phi,Y,ridge)
    for _ in range(n_iter):
        for j in range(Xi.shape[1]):
            idx=np.flatnonzero(np.abs(Xi[:,j])>=threshold)
            if len(idx)>max_terms: idx=idx[np.argsort(np.abs(Xi[idx,j]))[-max_terms:]]
            col=np.zeros(Phi.shape[1]);
            if len(idx): col[idx]=_ridge(Phi[:,idx],Y[:,j],ridge)
            Xi[:,j]=col
    return Xi

def _targets(X,Xn,dt):
    th=X[:,2]; dx=(Xn[:,0]-X[:,0])/dt; dy=(Xn[:,1]-X[:,1])/dt; slip=-np.sin(th)*dx+np.cos(th)*dy
    return np.column_stack([Xn[:,3],Xn[:,4],slip])

def _confidence(gain,recent,norm_id,nnz,max_terms):
    fit=np.exp(-norm_id/0.20); g=np.clip(0.5+gain/0.12,0,1); r=np.clip(0.5+recent/0.10,0,1); sparse=np.clip(1-nnz/max(1.0,3*max_terms),0,1)
    return float(np.clip(0.25*fit+0.40*g+0.25*r+0.10*sparse,0,1))

def _blend_state(nom,learn,alpha): return (1-alpha)*nom+alpha*learn

def _multistep_alpha(X0,Uv,Xtrue,dt,Xi,confidence,prev_alpha,mode,smoothing,amin,amax,fixed_alpha=0.45):
    if mode=='fixed': return float(np.clip(fixed_alpha,0,1))
    H=min(5,len(Uv)-1); candidates=np.linspace(0,0.40,9); costs=[]
    starts=range(0,max(1,len(Uv)-H))
    for a in candidates:
        cost=0.0; count=0
        for j in starts:
            xn=X0[j].copy()
            for h in range(H):
                nom=nominal_dynamics(xn,Uv[j+h],dt)
                phi,_=build_feature_matrix(xn[None,:],Uv[j+h][None,:]); p=(phi@Xi).reshape(-1); th=xn[2]; learn=xn.copy(); learn[3]=p[0]; learn[4]=p[1]
                learn[0]=xn[0]+(p[0]*np.cos(th)-p[2]*np.sin(th))*dt; learn[1]=xn[1]+(p[0]*np.sin(th)+p[2]*np.cos(th))*dt; learn[2]=wrap_angle(th+p[1]*dt)
                xn=_blend_state(nom,learn,a)
            cost += np.linalg.norm(xn[:2]-Xtrue[j+H-1,:2]) + 0.35*abs(wrap_angle(xn[2]-Xtrue[j+H-1,2])); count+=1
        costs.append(cost/max(count,1))
    best_idx=int(np.argmin(costs)); a_star=float(candidates[best_idx])
    ref_idx=int(np.argmin(np.abs(candidates-0.10))); ref_cost=float(costs[ref_idx])
    improvement=(ref_cost-float(costs[best_idx]))/(abs(ref_cost)+1e-12)
    if improvement < 0.02:
        a_star=0.10
    target=a_star*(0.45+0.55*confidence)
    return float(np.clip((1-smoothing)*prev_alpha+smoothing*target,amin,amax))

def update_model(states,inputs,next_states,dt,p,prev_model=None):
    X=np.asarray(states,float); U=np.asarray(inputs,float); Xn=np.asarray(next_states,float); n=len(X)
    if n<p.min_samples: return None,'too_few_samples'
    # Regime-aware fast reset: once a causal alarm is raised, identify from a
    # recent-only window so pre-change samples do not anchor the post-change model.
    if prev_model is not None and prev_model.regime_alarm>0.5 and n>=24:
        keep=min(24,n); X=X[-keep:]; U=U[-keep:]; Xn=Xn[-keep:]; n=len(X)
    Phi,names=build_feature_matrix(X,U); Y=_targets(X,Xn,dt); split=max(12,int((1-p.validation_ratio)*n)); split=min(split,n-6)
    Xi=stlsq(Phi[:split],Y[:split],p.threshold,p.ridge,p.stlsq_iters,p.max_terms_per_channel)
    # Temporal support stabilization: retain a small amount of the previous
    # identified model to suppress coefficient chattering across updates.
    if prev_model is not None and getattr(prev_model, 'Xi', None) is not None and prev_model.Xi.shape == Xi.shape:
        Xi = 0.78 * Xi + 0.22 * prev_model.Xi
        Xi[np.abs(Xi) < 0.55 * p.threshold] = 0.0
    Xv=X[split:]; Uv=U[split:]; Xnv=Xn[split:]
    pred=[]; nom=[]
    for xx,uu in zip(Xv,Uv):
        nom.append(nominal_dynamics(xx,uu,dt)); phi,_=build_feature_matrix(xx[None,:],uu[None,:]); z=(phi@Xi).reshape(-1); th=xx[2]; q=xx.copy(); q[3]=z[0]; q[4]=z[1]; q[0]=xx[0]+(z[0]*np.cos(th)-z[2]*np.sin(th))*dt; q[1]=xx[1]+(z[0]*np.sin(th)+z[2]*np.cos(th))*dt; q[2]=wrap_angle(th+z[1]*dt); pred.append(q)
    pred=np.asarray(pred); nom=np.asarray(nom); en=np.linalg.norm(nom-Xnv,axis=1); el=np.linalg.norm(pred-Xnv,axis=1)
    gain=float(1-np.mean(el)/(np.mean(en)+1e-12)); hs=slice(len(el)//2,None); recent=float(1-np.mean(el[hs])/(np.mean(en[hs])+1e-12))
    norm_id=float(np.mean(np.linalg.norm(Phi@Xi-Y,axis=1))/(np.mean(np.linalg.norm(Y,axis=1))+1e-12)); nnz=int(np.count_nonzero(np.abs(Xi)>1e-9)); conf=_confidence(gain,recent,norm_id,nnz,p.max_terms_per_channel)
    prev_alpha=float(prev_model.alpha if prev_model else 0.0); mode=getattr(p,'alpha_mode','adaptive')
    alpha=_multistep_alpha(Xv,Uv,Xnv,dt,Xi,conf,prev_alpha,mode,p.alpha_smoothing,p.alpha_min,p.alpha_max,p.fixed_alpha)
    ok=(gain>=p.min_prediction_gain and recent>=p.min_recent_gain and norm_id<=p.max_norm_id_error and conf>=p.confidence_min and nnz<=3*p.max_terms_per_channel)
    if not ok: return None,f'quality_gate:g={gain:.3f},r={recent:.3f},id={norm_id:.3f},conf={conf:.3f},nnz={nnz}'
    center=np.mean(Phi[:split],axis=0)
    scale=np.std(Phi[:split],axis=0)
    mdl=SINDyModel(Xi,names,conf,alpha,gain,recent,norm_id,nnz,split,
                      (prev_model.version+1 if prev_model else 1), 'sindy', center, scale,
                      3.0,
                      learned_resid_ema=(prev_model.learned_resid_ema if prev_model else np.nan),
                      nominal_resid_ema=(prev_model.nominal_resid_ema if prev_model else np.nan),
                      regime_score=(prev_model.regime_score if prev_model else 0.0),
                      regime_alarm=(prev_model.regime_alarm if prev_model else 0.0),
                      alpha_reset=0.12,
                      learned_ref_ema=(prev_model.learned_ref_ema if prev_model else np.nan),
                      regime_count=(prev_model.regime_count if prev_model else 0),
                      local_validity_ema=(prev_model.local_validity_ema if prev_model else 1.0),
                      local_advantage_ema=(prev_model.local_advantage_ema if prev_model else np.nan))
    return mdl, 'accepted'
