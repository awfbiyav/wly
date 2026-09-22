from pathlib import Path
import numpy as np, pandas as pd
from ..config.scenarios import Scenario, Obstacle
from ..dynamics.true import true_step
from ..dynamics.base import nominal_dynamics, wrap_angle
from ..sindy.library import build_feature_matrix
from ..sindy.model import stlsq, _targets

SEED=74011

def scenarios(seed=SEED,n=6):
    rng=np.random.default_rng(seed); out=[]
    for i in range(n):
        st=float(rng.uniform(7.0,8.5)); pv=float(rng.uniform(.78,.92)); pw=float(rng.uniform(.75,.92)); ps=float(rng.uniform(.04,.09))
        qv=float(rng.uniform(.42,.62)); qw=float(rng.uniform(.25,.45)); qs=float(rng.uniform(.22,.36))
        out.append(Scenario(f'PB{i+1}',i%5,Obstacle(np.array([30.,30.]),.2),np.array([9.,2.]),22.,False,pv,pw,ps,False,np.zeros(2),.02,0.,0.,st,qv,qw,qs,True))
    return out

def truth_roll(scene,seed,T=22.,dt=.1):
    rng=np.random.default_rng(seed); n=int(T/dt); x=np.zeros(5); rows=[]
    for k in range(n):
        t=k*dt; u=np.array([0.75+0.25*np.sin(.45*t+.3), .65*np.sin(.72*t+.7)+.18*np.sin(1.1*t)])
        xn=true_step(x,u,dt,scene,t,rng); rows.append((t,x.copy(),u.copy(),xn.copy())); x=xn
    return rows

def predict_next_param(x,u,dt,Xi):
    phi,_=build_feature_matrix(np.asarray(x)[None,:],np.asarray(u)[None,:]); z=(phi@Xi).reshape(-1); th=x[2]; q=x.copy(); q[3]=z[0]; q[4]=z[1]; q[0]=x[0]+(z[0]*np.cos(th)-z[2]*np.sin(th))*dt; q[1]=x[1]+(z[0]*np.sin(th)+z[2]*np.cos(th))*dt; q[2]=wrap_angle(th+z[1]*dt); return q

def run(seeds=range(421,441), out=None):
    rows=[]; scs=scenarios()
    for si,seed in enumerate(seeds,1):
        for scene in scs:
            data=truth_roll(scene,int(seed)); split=int(scene.regime_shift_time/.1); post_start=min(len(data)-1,split+5); 
            X=np.array([r[1] for r in data[:split]]); U=np.array([r[2] for r in data[:split]]); Xn=np.array([r[3] for r in data[:split]])
            Phi,_=build_feature_matrix(X,U); Y=_targets(X,Xn,.1)
            # Pre-shift SINDy model, then post-shift refit every 5 samples using the recent window.
            sindy_err=[]; nom_err=[]; sparse_nnz=[]; rls_err=[]; post_gain=[]
            # Simple RLS on the same library, one output at a time.
            d=Phi.shape[1]; P=[100*np.eye(d) for _ in range(3)]; th=[np.zeros(d) for _ in range(3)]
            lam=.995
            for yy,ph in zip(Y,Phi):
                for j in range(3):
                    Pj=P[j]; tj=th[j]; den=lam+ph@Pj@ph; K=Pj@ph/den; tj=tj+K*(yy[j]-ph@tj); P[j]=(Pj-np.outer(K,ph@Pj))/lam; th[j]=tj
            Xi=stlsq(Phi,Y,.025,1e-5,8,5)
            for k in range(post_start,len(data)):
                # periodic recent refit after shift
                if (k-post_start)%5==0 and k>=25:
                    recent=data[max(0,k-45):k]; XX=np.array([r[1] for r in recent]); UU=np.array([r[2] for r in recent]); XXn=np.array([r[3] for r in recent]); PP,_=build_feature_matrix(XX,UU); YY=_targets(XX,XXn,.1); Xi=stlsq(PP,YY,.025,1e-5,8,5)
                _,x,u,xn=data[k]
                ps=predict_next_param(x,u,.1,Xi); pn=nominal_dynamics(x,u,.1)
                phi,_=build_feature_matrix(np.asarray(x)[None,:],np.asarray(u)[None,:]); zr=np.array([phi@th[j] for j in range(3)]).reshape(3); thx=x[2]; pr=x.copy(); pr[3]=zr[0]; pr[4]=zr[1]; pr[0]=x[0]+(zr[0]*np.cos(thx)-zr[2]*np.sin(thx))*.1; pr[1]=x[1]+(zr[0]*np.sin(thx)+zr[2]*np.cos(thx))*.1; pr[2]=wrap_angle(thx+zr[1]*.1)
                es=np.linalg.norm(ps-xn); en=np.linalg.norm(pn-xn); er=np.linalg.norm(pr-xn)
                sindy_err.append(es); nom_err.append(en); rls_err.append(er); sparse_nnz.append(np.count_nonzero(np.abs(Xi)>1e-9))
            rows.append({'seed':int(seed),'scenario':scene.name,'sindy_rmse':float(np.sqrt(np.mean(np.array(sindy_err)**2))),'nominal_rmse':float(np.sqrt(np.mean(np.array(nom_err)**2))),'rls_rmse':float(np.sqrt(np.mean(np.array(rls_err)**2))),'sindy_gain_vs_nominal':float(1-np.mean(sindy_err)/(np.mean(nom_err)+1e-12)),'sindy_gain_vs_rls':float(1-np.mean(sindy_err)/(np.mean(rls_err)+1e-12)),'nnz':float(np.mean(sparse_nnz))})
        print(f'[pred] {si}/{len(list(seeds))} seeds done',flush=True)
    df=pd.DataFrame(rows)
    if out: Path(out).parent.mkdir(parents=True,exist_ok=True); df.to_csv(out,index=False)
    return df

if __name__=='__main__':
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else 'results/prediction_benchmark_rerun.csv'
    df=run(out=out); print(df[['sindy_rmse','nominal_rmse','rls_rmse','sindy_gain_vs_nominal','sindy_gain_vs_rls']].mean())
